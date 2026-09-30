"""问题3：基于储能时移的多目标能源协同优化模型。

模型口径与《q3模型.docx》保持一致：

* 0--2405小时为可调度运行时段，2406小时只作为终端状态结算；
* IT负荷固定为 ``Baseline_AI_IT_Load_MW + NonAI_IT_Load_MW``，不再优化任务迁移；
* 新能源分为直接消纳、储能充电、外送和弃用；电网购电可以直接供负荷或给储能充电；
* 储能按区域独立运行，SOC、充放电功率和充放电互斥约束形成MILP；
* 先分别求解成本、碳排放、峰值净购电和净购电爬坡量四个单目标锚点，
  再进行最小最大偏离、偏离和、储能吞吐量三级均衡择优；
* 结果写入本题 ``outputs/tables``，不读取其他题目的模型结果。

求解器使用项目已有的 ``scipy.optimize.milp``，本文件不新增第三方依赖。
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


QUESTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = QUESTION_DIR.parents[1]
PROCESSED_DIR = QUESTION_DIR / "data" / "processed"
Q3_DIR = PROCESSED_DIR / "q3"
SHARED_DIR = PROJECT_DIR / "question" / "question_01" / "data" / "processed" / "shared"
TABLES_DIR = QUESTION_DIR / "outputs" / "tables"
LOGS_DIR = QUESTION_DIR / "outputs" / "logs"
MODEL_LOG_PATH = LOGS_DIR / "question_03_model.log"

MAIN_START_HOUR = 0
FULL_OPERATIONAL_END_HOUR = 2405
FULL_TERMINAL_HOUR = 2406
FLOAT_EPS = 1e-8
AUDIT_TOLERANCE = 1e-5
BALANCE_TOLERANCE = 1e-6
DEFAULT_SOLVER_PROGRESS_INTERVAL = 30.0
OBJECTIVE_NAMES = ("Cost", "Carbon", "Peak", "Ramp")

Q3_INPUT_COLUMNS = (
    "Hour",
    "Region",
    "TimeRole",
    "Fixed_Facility_Load_MW",
    "AvailableRenewable_MW",
    "ElectricityPrice_CNY_per_MWh",
    "SellPrice_CNY_per_MWh",
    "CarbonIntensity_tCO2_per_MWh",
)
STORAGE_COLUMNS = (
    "Region",
    "StorageCapacity_MWh",
    "MinSOC_MWh",
    "InitialSOC_MWh",
    "MaxChargePower_MW",
    "MaxDischargePower_MW",
    "ChargeEfficiency",
    "DischargeEfficiency",
    "SellLimit_MW",
    "MaxGridImport_MW",
    "MaxGridExport_MW",
)
BASELINE_COLUMNS = (
    "Hour",
    "Region",
    "UsedRenewable_MW",
    "RenewableCharge_MW",
    "Curtailment_MW",
    "GridPurchase_MW",
    "GridCharge_MW",
    "GridSell_MW",
    "NetGridImport_MW",
    "SOC_MWh",
    "ChargePower_MW",
    "DischargePower_MW",
)


@dataclass(frozen=True)
class Q3Data:
    """已经对齐到Hour×Region键的Q3运行数据。"""

    operating: pd.DataFrame
    terminal: pd.DataFrame
    baseline_operating: pd.DataFrame
    baseline_terminal: pd.DataFrame
    regions: tuple[str, ...]
    hours: tuple[int, ...]
    storage: pd.DataFrame


@dataclass(frozen=True)
class VariableIndex:
    renewable_charge: np.ndarray
    grid_charge: np.ndarray
    discharge: np.ndarray
    direct_renewable: np.ndarray
    grid_purchase: np.ndarray
    renewable_export: np.ndarray
    curtailment: np.ndarray
    soc: np.ndarray
    charging_state: np.ndarray
    ramp: np.ndarray
    peak: np.ndarray
    z: int
    count: int


@dataclass(frozen=True)
class LinearModel:
    matrix: object
    lower: np.ndarray
    upper: np.ndarray
    variable_lower: np.ndarray
    variable_upper: np.ndarray
    integrality: np.ndarray
    index: VariableIndex
    objective_vectors: dict[str, np.ndarray]
    throughput_vector: np.ndarray
    data_arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class Solution:
    vector: np.ndarray
    status: int
    success: bool
    message: str
    elapsed_seconds: float
    mip_gap: float
    mip_dual_bound: float


@dataclass(frozen=True)
class DispatchArrays:
    direct_renewable: np.ndarray
    renewable_charge: np.ndarray
    grid_charge: np.ndarray
    discharge: np.ndarray
    grid_purchase: np.ndarray
    renewable_export: np.ndarray
    curtailment: np.ndarray
    soc: np.ndarray


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"不支持的日志级别：{level_name}")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(MODEL_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(
        level=level,
        handlers=[stream_handler, file_handler],
        force=True,
    )


def _read_csv(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少模型输入文件：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}缺少字段：{missing}")
    return frame


def _to_numeric(frame: pd.DataFrame, columns: Iterable[str], source_name: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any() or not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{source_name}的{column}存在无效数值")
    return result


def _validate_hour_region(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    result = frame.copy()
    if result[["Hour", "Region"]].isna().any().any():
        raise ValueError(f"{source_name}的Hour或Region存在缺失值")
    result["Hour"] = pd.to_numeric(result["Hour"], errors="coerce")
    if result["Hour"].isna().any() or not result["Hour"].eq(result["Hour"].round()).all():
        raise ValueError(f"{source_name}的Hour不是有效整数")
    result["Hour"] = result["Hour"].astype("int64")
    result["Region"] = result["Region"].astype(str)
    if result.duplicated(["Hour", "Region"]).any():
        raise ValueError(f"{source_name}存在重复的Hour×Region记录")
    return result


def _validate_complete_grid(
    frame: pd.DataFrame,
    regions: tuple[str, ...],
    first_hour: int,
    last_hour: int,
    source_name: str,
) -> None:
    expected = {(hour, region) for hour in range(first_hour, last_hour + 1) for region in regions}
    actual = set(zip(frame["Hour"].astype(int), frame["Region"].astype(str)))
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        sample_missing = sorted(missing)[:5]
        sample_extra = sorted(extra)[:5]
        raise ValueError(
            f"{source_name}的Hour×Region网格不完整；缺失示例={sample_missing}，多余示例={sample_extra}"
        )


def _merge_storage(frame: pd.DataFrame, storage: pd.DataFrame, source_name: str) -> pd.DataFrame:
    result = frame.merge(storage, how="left", on="Region", validate="many_to_one")
    if result[list(STORAGE_COLUMNS[1:])].isna().any().any():
        raise ValueError(f"{source_name}无法被storage_params完整覆盖")
    return result


def _load_data(end_hour: int) -> Q3Data:
    q3_input = _read_csv(Q3_DIR / "q3_fixed_energy_input.csv", Q3_INPUT_COLUMNS)
    q3_input = _validate_hour_region(q3_input, "q3_fixed_energy_input.csv")
    q3_input = _to_numeric(
        q3_input,
        [
            "Hour",
            "Fixed_Facility_Load_MW",
            "AvailableRenewable_MW",
            "ElectricityPrice_CNY_per_MWh",
            "SellPrice_CNY_per_MWh",
            "CarbonIntensity_tCO2_per_MWh",
        ],
        "q3_fixed_energy_input.csv",
    )
    if (q3_input["Fixed_Facility_Load_MW"] < 0).any() or (q3_input["AvailableRenewable_MW"] < 0).any():
        raise ValueError("Q3固定设施负荷和可用新能源不能为负")

    regions = tuple(sorted(q3_input["Region"].unique().tolist()))
    if not regions:
        raise ValueError("Q3没有可用区域")
    terminal_hour = end_hour + 1
    q3_input = q3_input[q3_input["Hour"] <= terminal_hour].copy()
    _validate_complete_grid(q3_input, regions, MAIN_START_HOUR, terminal_hour, "q3_fixed_energy_input.csv")
    q3_input = q3_input.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)

    storage = _read_csv(SHARED_DIR / "storage_params.csv", STORAGE_COLUMNS)
    storage["Region"] = storage["Region"].astype(str)
    if storage["Region"].duplicated().any() or set(storage["Region"]) != set(regions):
        raise ValueError("storage_params.csv的区域集合必须与Q3输入一致且唯一")
    storage = _to_numeric(storage, STORAGE_COLUMNS[1:], "storage_params.csv")
    numeric_storage = storage.loc[:, list(STORAGE_COLUMNS[1:])]
    if (numeric_storage < 0).any().any():
        raise ValueError("storage_params.csv不能包含负的容量、功率或边界")
    if (storage["StorageCapacity_MWh"] < storage["MinSOC_MWh"]).any():
        raise ValueError("储能最小SOC不能高于容量")
    if (
        (storage["InitialSOC_MWh"] < storage["MinSOC_MWh"])
        | (storage["InitialSOC_MWh"] > storage["StorageCapacity_MWh"])
    ).any():
        raise ValueError("初始SOC必须位于储能SOC边界内")
    if (
        (storage["ChargeEfficiency"] <= 0)
        | (storage["ChargeEfficiency"] > 1)
        | (storage["DischargeEfficiency"] <= 0)
        | (storage["DischargeEfficiency"] > 1)
    ).any():
        raise ValueError("充放电效率必须位于(0,1]")
    storage = storage.sort_values("Region", kind="stable").reset_index(drop=True)

    baseline = _read_csv(SHARED_DIR / "baseline_reference_region_hour.csv", BASELINE_COLUMNS)
    baseline = _validate_hour_region(baseline, "baseline_reference_region_hour.csv")
    baseline = _to_numeric(
        baseline,
        [column for column in BASELINE_COLUMNS if column not in ("Region",)],
        "baseline_reference_region_hour.csv",
    )
    baseline = baseline[baseline["Hour"] <= terminal_hour].copy()
    _validate_complete_grid(
        baseline,
        regions,
        MAIN_START_HOUR,
        terminal_hour,
        "baseline_reference_region_hour.csv",
    )

    operation = q3_input[q3_input["Hour"] <= end_hour].copy()
    terminal = q3_input[q3_input["Hour"] == terminal_hour].copy()
    baseline_operating = baseline[baseline["Hour"] <= end_hour].copy()
    baseline_terminal = baseline[baseline["Hour"] == terminal_hour].copy()
    operation = _merge_storage(operation, storage, "q3_fixed_energy_input.csv")
    terminal = _merge_storage(terminal, storage, "q3_fixed_energy_input.csv terminal")

    _validate_complete_grid(operation, regions, MAIN_START_HOUR, end_hour, "Q3运行输入")
    _validate_complete_grid(baseline_operating, regions, MAIN_START_HOUR, end_hour, "Q3基准运行状态")
    operation = operation.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    terminal = terminal.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    baseline_operating = baseline_operating.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    baseline_terminal = baseline_terminal.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)

    logging.info(
        "输入读取完成：运行时段=%d--%d，共%d小时；区域=%s；终端小时=%d。",
        MAIN_START_HOUR,
        end_hour,
        end_hour - MAIN_START_HOUR + 1,
        ",".join(regions),
        terminal_hour,
    )
    return Q3Data(
        operating=operation,
        terminal=terminal,
        baseline_operating=baseline_operating,
        baseline_terminal=baseline_terminal,
        regions=regions,
        hours=tuple(range(MAIN_START_HOUR, end_hour + 1)),
        storage=storage,
    )


def _block_indices(offset: int, shape: tuple[int, ...]) -> tuple[np.ndarray, int]:
    size = int(np.prod(shape))
    return np.arange(offset, offset + size, dtype=int).reshape(shape), offset + size


def _make_indices(time_count: int, region_count: int) -> VariableIndex:
    offset = 0
    renewable_charge, offset = _block_indices(offset, (time_count, region_count))
    grid_charge, offset = _block_indices(offset, (time_count, region_count))
    discharge, offset = _block_indices(offset, (time_count, region_count))
    direct_renewable, offset = _block_indices(offset, (time_count, region_count))
    grid_purchase, offset = _block_indices(offset, (time_count, region_count))
    renewable_export, offset = _block_indices(offset, (time_count, region_count))
    curtailment, offset = _block_indices(offset, (time_count, region_count))
    soc, offset = _block_indices(offset, (time_count + 1, region_count))
    charging_state, offset = _block_indices(offset, (time_count, region_count))
    ramp, offset = _block_indices(offset, (max(time_count - 1, 0), region_count))
    peak, offset = _block_indices(offset, (region_count,))
    z = offset
    return VariableIndex(
        renewable_charge=renewable_charge,
        grid_charge=grid_charge,
        discharge=discharge,
        direct_renewable=direct_renewable,
        grid_purchase=grid_purchase,
        renewable_export=renewable_export,
        curtailment=curtailment,
        soc=soc,
        charging_state=charging_state,
        ramp=ramp,
        peak=peak,
        z=z,
        count=z + 1,
    )


def _rows_to_sparse(rows: list[dict[int, float]], variable_count: int):
    from scipy.sparse import coo_matrix

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, terms in enumerate(rows):
        for column, value in terms.items():
            if abs(value) <= FLOAT_EPS:
                continue
            row_indices.append(row_index)
            column_indices.append(int(column))
            values.append(float(value))
    return coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(rows), variable_count),
    ).tocsr()


def _build_linear_model(data: Q3Data) -> LinearModel:
    from scipy.sparse import csr_matrix

    frame = data.operating
    time_count = len(data.hours)
    region_count = len(data.regions)
    if len(frame) != time_count * region_count:
        raise ValueError("Q3运行输入行数与时间×区域网格不一致")

    index = _make_indices(time_count, region_count)
    variable_lower = np.zeros(index.count, dtype=float)
    variable_upper = np.full(index.count, np.inf, dtype=float)
    integrality = np.zeros(index.count, dtype=np.int8)
    integrality[index.charging_state.ravel()] = 1
    variable_upper[index.charging_state.ravel()] = 1.0

    ordered = frame.sort_values(["Hour", "Region"], kind="stable")
    arrays = {
        column: ordered[column].to_numpy(dtype=float).reshape(time_count, region_count)
        for column in (
            "Fixed_Facility_Load_MW",
            "AvailableRenewable_MW",
            "ElectricityPrice_CNY_per_MWh",
            "SellPrice_CNY_per_MWh",
            "CarbonIntensity_tCO2_per_MWh",
        )
    }
    storage = data.storage.set_index("Region").loc[list(data.regions)]
    storage_arrays = {
        column: storage[column].to_numpy(dtype=float)
        for column in STORAGE_COLUMNS[1:]
    }
    arrays.update(storage_arrays)

    fixed_load = arrays["Fixed_Facility_Load_MW"]
    renewable = arrays["AvailableRenewable_MW"]
    max_charge = arrays["MaxChargePower_MW"]
    max_discharge = arrays["MaxDischargePower_MW"]
    max_grid_import = arrays["MaxGridImport_MW"]
    max_export = np.maximum(arrays["SellLimit_MW"], arrays["MaxGridExport_MW"])

    variable_upper[index.renewable_charge.ravel()] = np.tile(max_charge, time_count)
    variable_upper[index.grid_charge.ravel()] = np.tile(max_charge, time_count)
    variable_upper[index.discharge.ravel()] = np.tile(max_discharge, time_count)
    variable_upper[index.direct_renewable.ravel()] = np.minimum(fixed_load, renewable).ravel()
    variable_upper[index.grid_purchase.ravel()] = np.tile(max_grid_import, time_count)
    variable_upper[index.renewable_export.ravel()] = np.tile(max_export, time_count)
    variable_upper[index.curtailment.ravel()] = renewable.ravel()
    variable_lower[index.soc.ravel()] = np.tile(arrays["MinSOC_MWh"], time_count + 1)
    variable_upper[index.soc.ravel()] = np.tile(arrays["StorageCapacity_MWh"], time_count + 1)
    variable_upper[index.ramp.ravel()] = np.tile(max_grid_import + max_export, max(time_count - 1, 0))
    variable_upper[index.peak.ravel()] = max_grid_import
    variable_upper[index.z] = 1e6

    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_row(terms: dict[int, float], low: float = -np.inf, high: float = np.inf) -> None:
        rows.append(terms)
        lower.append(float(low))
        upper.append(float(high))

    for t in range(time_count):
        for r in range(region_count):
            cr = int(index.renewable_charge[t, r])
            cg = int(index.grid_charge[t, r])
            d = int(index.discharge[t, r])
            v = int(index.direct_renewable[t, r])
            b = int(index.grid_purchase[t, r])
            x = int(index.renewable_export[t, r])
            curtail = int(index.curtailment[t, r])
            state_now = int(index.soc[t, r])
            state_next = int(index.soc[t + 1, r])
            y = int(index.charging_state[t, r])
            charge_limit = float(max_charge[r])
            discharge_limit = float(max_discharge[r])
            charge_efficiency = float(arrays["ChargeEfficiency"][r])
            discharge_efficiency = float(arrays["DischargeEfficiency"][r])

            # 可用新能源全部进入直接消纳、储能充电、外送或弃用。
            add_row(
                {v: 1.0, cr: 1.0, x: 1.0, curtail: 1.0},
                float(renewable[t, r]),
                float(renewable[t, r]),
            )
            # 电网购电包含给储能的电网充电；新能源充电已在新能源平衡中单独结算。
            add_row(
                {b: 1.0, v: 1.0, d: 1.0, cg: -1.0},
                float(fixed_load[t, r]),
                float(fixed_load[t, r]),
            )
            add_row(
                {
                    state_next: 1.0,
                    state_now: -1.0,
                    cr: -charge_efficiency,
                    cg: -charge_efficiency,
                    d: 1.0 / discharge_efficiency,
                },
                0.0,
                0.0,
            )
            add_row({cr: 1.0, cg: 1.0, y: -charge_limit}, -np.inf, 0.0)
            add_row({d: 1.0, y: discharge_limit}, -np.inf, discharge_limit)
            add_row({cg: 1.0, b: -1.0}, -np.inf, 0.0)
            add_row({v: 1.0}, -np.inf, float(fixed_load[t, r]))
            add_row({v: 1.0}, -np.inf, float(renewable[t, r]))
            # 两个外送边界保持为两个独立约束，便于结果审计和论文解释。
            add_row({x: 1.0}, -np.inf, float(arrays["SellLimit_MW"][r]))
            add_row({x: 1.0}, -np.inf, float(arrays["MaxGridExport_MW"][r]))
            p = int(index.peak[r])
            add_row({p: 1.0, b: -1.0, x: 1.0}, 0.0, np.inf)

            if t < time_count - 1:
                b_next = int(index.grid_purchase[t + 1, r])
                x_next = int(index.renewable_export[t + 1, r])
                w = int(index.ramp[t, r])
                # W >= N(t+1)-N(t)。
                add_row({w: 1.0, b_next: -1.0, x_next: 1.0, b: 1.0, x: -1.0}, 0.0, np.inf)
                # W >= N(t)-N(t+1)。
                add_row({w: 1.0, b: -1.0, x: 1.0, b_next: 1.0, x_next: -1.0}, 0.0, np.inf)

    for r in range(region_count):
        state_initial = int(index.soc[0, r])
        state_terminal = int(index.soc[time_count, r])
        initial_soc = float(arrays["InitialSOC_MWh"][r])
        add_row({state_initial: 1.0}, initial_soc, initial_soc)
        add_row({state_terminal: 1.0}, initial_soc, np.inf)

    objective_vectors = {
        "Cost": np.zeros(index.count, dtype=float),
        "Carbon": np.zeros(index.count, dtype=float),
        "Peak": np.zeros(index.count, dtype=float),
        "Ramp": np.zeros(index.count, dtype=float),
    }
    throughput_vector = np.zeros(index.count, dtype=float)
    price = arrays["ElectricityPrice_CNY_per_MWh"]
    sell_price = arrays["SellPrice_CNY_per_MWh"]
    carbon = arrays["CarbonIntensity_tCO2_per_MWh"]
    objective_vectors["Cost"][index.grid_purchase.ravel()] = price.ravel()
    objective_vectors["Cost"][index.renewable_export.ravel()] = -sell_price.ravel()
    objective_vectors["Carbon"][index.grid_purchase.ravel()] = carbon.ravel()
    objective_vectors["Peak"][index.peak.ravel()] = 1.0
    if time_count > 1:
        objective_vectors["Ramp"][index.ramp.ravel()] = 1.0 / (region_count * (time_count - 1))
    throughput_vector[index.renewable_charge.ravel()] = 1.0
    throughput_vector[index.grid_charge.ravel()] = 1.0
    throughput_vector[index.discharge.ravel()] = 1.0

    matrix = _rows_to_sparse(rows, index.count)
    if not isinstance(matrix, csr_matrix):
        matrix = matrix.tocsr()
    logging.info(
        "MILP结构建立完成：变量=%d（含二元变量=%d），基础约束=%d，非零元=%d。",
        index.count,
        int(integrality.sum()),
        matrix.shape[0],
        matrix.nnz,
    )
    return LinearModel(
        matrix=matrix,
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
        variable_lower=variable_lower,
        variable_upper=variable_upper,
        integrality=integrality,
        index=index,
        objective_vectors=objective_vectors,
        throughput_vector=throughput_vector,
        data_arrays=arrays,
    )


def _append_constraints(
    model: LinearModel,
    extra_rows: list[tuple[dict[int, float], float, float]],
):
    from scipy.sparse import vstack

    if not extra_rows:
        return model.matrix, model.lower, model.upper
    extra_matrix = _rows_to_sparse(
        [row[0] for row in extra_rows],
        model.index.count,
    )
    matrix = vstack([model.matrix, extra_matrix], format="csr")
    lower = np.concatenate(
        [model.lower, np.asarray([row[1] for row in extra_rows], dtype=float)]
    )
    upper = np.concatenate(
        [model.upper, np.asarray([row[2] for row in extra_rows], dtype=float)]
    )
    return matrix, lower, upper


def _result_float(result: object, field: str) -> float:
    value = getattr(result, field, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _solve(
    model: LinearModel,
    objective: np.ndarray,
    stage_name: str,
    extra_rows: list[tuple[dict[int, float], float, float]] | None = None,
    mip_rel_gap: float = 1e-5,
    progress_interval: float = DEFAULT_SOLVER_PROGRESS_INTERVAL,
    solver_disp: bool = False,
) -> Solution:
    from scipy.optimize import Bounds, LinearConstraint, milp

    matrix, lower, upper = _append_constraints(model, extra_rows or [])
    progress_interval = max(float(progress_interval), 0.1)
    logging.info(
        "%s开始：变量=%d，约束=%d（基础%d+额外%d），目标非零项=%d，mip_rel_gap=%g，心跳间隔=%.1fs。",
        stage_name,
        model.index.count,
        matrix.shape[0],
        model.matrix.shape[0],
        len(extra_rows or []),
        int(np.count_nonzero(np.abs(objective) > FLOAT_EPS)),
        mip_rel_gap,
        progress_interval,
    )
    started = time.perf_counter()
    heartbeat_stop = threading.Event()

    def _heartbeat() -> None:
        while not heartbeat_stop.wait(progress_interval):
            logging.info(
                "%s仍在求解：已耗时%.1fs，变量=%d，约束=%d；等待MILP返回。",
                stage_name,
                time.perf_counter() - started,
                model.index.count,
                matrix.shape[0],
            )

    heartbeat = threading.Thread(
        target=_heartbeat,
        name=f"q3-{stage_name}-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    try:
        result = milp(
            c=np.asarray(objective, dtype=float),
            integrality=model.integrality,
            bounds=Bounds(model.variable_lower, model.variable_upper),
            constraints=LinearConstraint(matrix, lower, upper),
            options={"disp": solver_disp, "mip_rel_gap": mip_rel_gap},
        )
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=1.0)
    elapsed = time.perf_counter() - started
    logging.info(
        "%s求解器返回：status=%s，success=%s，耗时%.2fs，mip_gap=%s，dual_bound=%s，message=%s。",
        stage_name,
        getattr(result, "status", "NA"),
        getattr(result, "success", "NA"),
        elapsed,
        _result_float(result, "mip_gap"),
        _result_float(result, "mip_dual_bound"),
        getattr(result, "message", "NA"),
    )
    vector = getattr(result, "x", None)
    if vector is None or not np.isfinite(np.asarray(vector, dtype=float)).all():
        raise RuntimeError(
            f"{stage_name}未返回可行解：status={result.status}，message={result.message}"
        )
    solution = Solution(
        vector=np.asarray(vector, dtype=float),
        status=int(result.status),
        success=bool(result.success),
        message=str(result.message),
        elapsed_seconds=elapsed,
        mip_gap=_result_float(result, "mip_gap"),
        mip_dual_bound=_result_float(result, "mip_dual_bound"),
    )
    if solution.status != 0:
        logging.warning(
            "%s返回可行但未确认最优的结果：status=%d，message=%s。",
            stage_name,
            solution.status,
            solution.message,
        )
    else:
        logging.info("%s完成：最优，耗时%.2fs。", stage_name, elapsed)
    return solution


def _metric_values(model: LinearModel, vector: np.ndarray) -> dict[str, float]:
    values = {
        name: float(np.dot(coefficients, vector))
        for name, coefficients in model.objective_vectors.items()
    }
    values["Throughput"] = float(np.dot(model.throughput_vector, vector))
    return values


def _vector_terms(vector: np.ndarray) -> dict[int, float]:
    return {
        int(index): float(value)
        for index, value in enumerate(vector)
        if abs(float(value)) > FLOAT_EPS
    }


def _normalized_rows(
    model: LinearModel,
    best: dict[str, float],
    scales: dict[str, float],
) -> list[tuple[dict[int, float], float, float]]:
    rows: list[tuple[dict[int, float], float, float]] = []
    for name in OBJECTIVE_NAMES:
        scale = scales[name]
        if scale <= FLOAT_EPS:
            continue
        terms = _vector_terms(model.objective_vectors[name])
        terms[model.index.z] = terms.get(model.index.z, 0.0) - scale
        # f_m - (f_m^ref-f_m^best)z <= f_m^best。
        rows.append((terms, -np.inf, best[name]))
    return rows


def _solve_multiobjective(
    model: LinearModel,
    mip_rel_gap: float,
    progress_interval: float = DEFAULT_SOLVER_PROGRESS_INTERVAL,
    solver_disp: bool = False,
) -> tuple[Solution, dict[str, Solution], dict[str, float], dict[str, float], list[dict[str, object]]]:
    anchor_solutions: dict[str, Solution] = {}
    anchor_metrics: dict[str, dict[str, float]] = {}
    solver_rows: list[dict[str, object]] = []

    for objective_name in OBJECTIVE_NAMES:
        stage = f"anchor_{objective_name}"
        logging.info("开始求解单目标锚点：%s。", objective_name)
        solution = _solve(
            model,
            model.objective_vectors[objective_name],
            stage,
            mip_rel_gap=mip_rel_gap,
            progress_interval=progress_interval,
            solver_disp=solver_disp,
        )
        metrics = _metric_values(model, solution.vector)
        anchor_solutions[objective_name] = solution
        anchor_metrics[objective_name] = metrics
        solver_rows.append(_solver_row(stage, solution, metrics, model))

    best = {
        name: float(anchor_metrics[name][name])
        for name in OBJECTIVE_NAMES
    }
    reference = {
        name: max(float(metrics[name]) for metrics in anchor_metrics.values())
        for name in OBJECTIVE_NAMES
    }
    scales = {
        name: max(0.0, reference[name] - best[name])
        for name in OBJECTIVE_NAMES
    }
    logging.info(
        "单目标锚点完成：best=%s，reference=%s。",
        {key: round(value, 8) for key, value in best.items()},
        {key: round(value, 8) for key, value in reference.items()},
    )

    stage_one_objective = np.zeros(model.index.count, dtype=float)
    stage_one_objective[model.index.z] = 1.0
    normalized_rows = _normalized_rows(model, best, scales)
    logging.info(
        "开始求解均衡阶段1：最小化最大归一化偏离；新增归一化约束=%d。",
        len(normalized_rows),
    )
    stage_one = _solve(
        model,
        stage_one_objective,
        "balanced_stage_1_minimax",
        extra_rows=normalized_rows,
        mip_rel_gap=mip_rel_gap,
        progress_interval=progress_interval,
        solver_disp=solver_disp,
    )
    stage_one_metrics = _metric_values(model, stage_one.vector)
    solver_rows.append(_solver_row("balanced_stage_1_minimax", stage_one, stage_one_metrics, model))
    z_star = float(stage_one.vector[model.index.z])
    z_upper = z_star + BALANCE_TOLERANCE * max(1.0, abs(z_star))
    z_row = ({model.index.z: 1.0}, -np.inf, z_upper)
    logging.info(
        "均衡阶段1完成：z*=%.10g；开始阶段2，固定z<=%.10g并最小化归一化偏离和。",
        z_star,
        z_upper,
    )

    stage_two_objective = np.zeros(model.index.count, dtype=float)
    active_scales = [name for name in OBJECTIVE_NAMES if scales[name] > FLOAT_EPS]
    for name in active_scales:
        stage_two_objective += model.objective_vectors[name] / scales[name]
    stage_two = _solve(
        model,
        stage_two_objective,
        "balanced_stage_2_sum_deviation",
        extra_rows=[*normalized_rows, z_row],
        mip_rel_gap=mip_rel_gap,
        progress_interval=progress_interval,
        solver_disp=solver_disp,
    )
    stage_two_metrics = _metric_values(model, stage_two.vector)
    solver_rows.append(_solver_row("balanced_stage_2_sum_deviation", stage_two, stage_two_metrics, model))
    sum_deviation = sum(
        (stage_two_metrics[name] - best[name]) / scales[name]
        for name in active_scales
    )
    sum_constant = sum(best[name] / scales[name] for name in active_scales)
    sum_terms = _vector_terms(stage_two_objective)
    sum_row = (
        sum_terms,
        -np.inf,
        sum_constant + sum_deviation + BALANCE_TOLERANCE * max(1.0, abs(sum_deviation)),
    )
    logging.info(
        "均衡阶段2完成：归一化偏离和=%.10g；开始阶段3，固定前两阶段水平并最小化储能吞吐量。",
        sum_deviation,
    )

    stage_three = _solve(
        model,
        model.throughput_vector,
        "balanced_stage_3_min_throughput",
        extra_rows=[*normalized_rows, z_row, sum_row],
        mip_rel_gap=mip_rel_gap,
        progress_interval=progress_interval,
        solver_disp=solver_disp,
    )
    stage_three_metrics = _metric_values(model, stage_three.vector)
    solver_rows.append(_solver_row("balanced_stage_3_min_throughput", stage_three, stage_three_metrics, model))
    logging.info(
        "均衡方案完成：z*=%.8g，归一化偏离和=%.8g，储能吞吐量=%.8g。",
        float(stage_three.vector[model.index.z]),
        sum(
            (stage_three_metrics[name] - best[name]) / scales[name]
            for name in active_scales
        ),
        stage_three_metrics["Throughput"],
    )
    return stage_three, anchor_solutions, best, reference, solver_rows


def _solver_row(
    stage_name: str,
    solution: Solution,
    metrics: dict[str, float],
    model: LinearModel,
) -> dict[str, object]:
    return {
        "Stage": stage_name,
        "Status": solution.status,
        "Success": solution.success,
        "Message": solution.message,
        "ElapsedSeconds": solution.elapsed_seconds,
        "MIPGap": solution.mip_gap,
        "MIPDualBound": solution.mip_dual_bound,
        "VariableCount": model.index.count,
        "BinaryVariableCount": int(model.integrality.sum()),
        "ConstraintCount": model.matrix.shape[0],
        "Cost": metrics["Cost"],
        "Carbon": metrics["Carbon"],
        "Peak": metrics["Peak"],
        "Ramp": metrics["Ramp"],
        "Throughput": metrics["Throughput"],
    }


def _solution_arrays(model: LinearModel, solution: Solution) -> DispatchArrays:
    vector = solution.vector
    index = model.index
    return DispatchArrays(
        direct_renewable=vector[index.direct_renewable].copy(),
        renewable_charge=vector[index.renewable_charge].copy(),
        grid_charge=vector[index.grid_charge].copy(),
        discharge=vector[index.discharge].copy(),
        grid_purchase=vector[index.grid_purchase].copy(),
        renewable_export=vector[index.renewable_export].copy(),
        curtailment=vector[index.curtailment].copy(),
        soc=vector[index.soc].copy(),
    )


def _baseline_arrays(data: Q3Data) -> DispatchArrays:
    frame = data.baseline_operating.sort_values(["Hour", "Region"], kind="stable")
    region_count = len(data.regions)
    time_count = len(data.hours)
    arrays = {
        column: frame[column].to_numpy(dtype=float).reshape(time_count, region_count)
        for column in BASELINE_COLUMNS[2:]
    }
    initial = data.storage.set_index("Region").loc[list(data.regions), "InitialSOC_MWh"].to_numpy(dtype=float)
    state = np.empty((time_count + 1, region_count), dtype=float)
    state[0] = initial
    state[1:] = arrays["SOC_MWh"]
    return DispatchArrays(
        direct_renewable=arrays["UsedRenewable_MW"],
        renewable_charge=arrays["RenewableCharge_MW"],
        grid_charge=arrays["GridCharge_MW"],
        discharge=arrays["DischargePower_MW"],
        grid_purchase=arrays["GridPurchase_MW"],
        renewable_export=arrays["GridSell_MW"],
        curtailment=arrays["Curtailment_MW"],
        soc=state,
    )


def _no_storage_arrays(data: Q3Data) -> DispatchArrays:
    frame = data.operating.sort_values(["Hour", "Region"], kind="stable")
    time_count = len(data.hours)
    region_count = len(data.regions)
    fixed_load = frame["Fixed_Facility_Load_MW"].to_numpy(dtype=float).reshape(time_count, region_count)
    renewable = frame["AvailableRenewable_MW"].to_numpy(dtype=float).reshape(time_count, region_count)
    sell_limit = data.storage.set_index("Region").loc[list(data.regions), "SellLimit_MW"].to_numpy(dtype=float)
    grid_export = data.storage.set_index("Region").loc[list(data.regions), "MaxGridExport_MW"].to_numpy(dtype=float)
    direct = np.minimum(fixed_load, renewable)
    grid_purchase = np.maximum(fixed_load - direct, 0.0)
    renewable_surplus = np.maximum(renewable - direct, 0.0)
    renewable_export = np.minimum(renewable_surplus, np.minimum(sell_limit, grid_export))
    curtailment = renewable_surplus - renewable_export
    initial = data.storage.set_index("Region").loc[list(data.regions), "InitialSOC_MWh"].to_numpy(dtype=float)
    state = np.repeat(initial.reshape(1, -1), time_count + 1, axis=0)
    zeros = np.zeros_like(direct)
    return DispatchArrays(
        direct_renewable=direct,
        renewable_charge=zeros.copy(),
        grid_charge=zeros.copy(),
        discharge=zeros.copy(),
        grid_purchase=grid_purchase,
        renewable_export=renewable_export,
        curtailment=curtailment,
        soc=state,
    )


def _metric_values_from_arrays(data: Q3Data, arrays: DispatchArrays) -> dict[str, float]:
    frame = data.operating.sort_values(["Hour", "Region"], kind="stable")
    time_count = len(data.hours)
    region_count = len(data.regions)
    price = frame["ElectricityPrice_CNY_per_MWh"].to_numpy(dtype=float).reshape(time_count, region_count)
    sell_price = frame["SellPrice_CNY_per_MWh"].to_numpy(dtype=float).reshape(time_count, region_count)
    carbon = frame["CarbonIntensity_tCO2_per_MWh"].to_numpy(dtype=float).reshape(time_count, region_count)
    net = arrays.grid_purchase - arrays.renewable_export
    return {
        "Cost": float(np.sum(price * arrays.grid_purchase - sell_price * arrays.renewable_export)),
        "Carbon": float(np.sum(carbon * arrays.grid_purchase)),
        "Peak": float(np.maximum(net, 0.0).max(axis=0).sum()),
        "Ramp": float(np.abs(np.diff(net, axis=0)).mean()) if time_count > 1 else 0.0,
        "Throughput": float(
            np.sum(arrays.renewable_charge + arrays.grid_charge + arrays.discharge)
        ),
    }


def _regional_metrics(data: Q3Data, scheme: str, arrays: DispatchArrays) -> pd.DataFrame:
    frame = data.operating.sort_values(["Hour", "Region"], kind="stable")
    time_count = len(data.hours)
    region_count = len(data.regions)
    price = frame["ElectricityPrice_CNY_per_MWh"].to_numpy(dtype=float).reshape(time_count, region_count)
    sell_price = frame["SellPrice_CNY_per_MWh"].to_numpy(dtype=float).reshape(time_count, region_count)
    carbon = frame["CarbonIntensity_tCO2_per_MWh"].to_numpy(dtype=float).reshape(time_count, region_count)
    net = arrays.grid_purchase - arrays.renewable_export
    terminal_soc = arrays.soc[-1]
    if scheme == "BaselineReference":
        terminal_soc = (
            data.baseline_terminal.sort_values("Region", kind="stable")
            .set_index("Region")
            .loc[list(data.regions), "SOC_MWh"]
            .to_numpy(dtype=float)
        )
    rows: list[dict[str, object]] = []
    for r, region in enumerate(data.regions):
        rows.append(
            {
                "Scheme": scheme,
                "Region": region,
                "Cost": float(np.sum(price[:, r] * arrays.grid_purchase[:, r] - sell_price[:, r] * arrays.renewable_export[:, r])),
                "Carbon": float(np.sum(carbon[:, r] * arrays.grid_purchase[:, r])),
                "Peak": float(np.maximum(net[:, r], 0.0).max()),
                "Ramp": float(np.abs(np.diff(net[:, r])).mean()) if time_count > 1 else 0.0,
                "Throughput": float(np.sum(arrays.renewable_charge[:, r] + arrays.grid_charge[:, r] + arrays.discharge[:, r])),
                "InitialSOC_MWh": float(arrays.soc[0, r]),
                "TerminalSOC_MWh": float(terminal_soc[r]),
            }
        )
    return pd.DataFrame(rows)


def _profile_frame(data: Q3Data, scheme: str, arrays: DispatchArrays) -> pd.DataFrame:
    frame = data.operating.sort_values(["Hour", "Region"], kind="stable").copy().reset_index(drop=True)
    price = frame["ElectricityPrice_CNY_per_MWh"].to_numpy(dtype=float)
    sell_price = frame["SellPrice_CNY_per_MWh"].to_numpy(dtype=float)
    carbon = frame["CarbonIntensity_tCO2_per_MWh"].to_numpy(dtype=float)
    direct = arrays.direct_renewable.reshape(-1)
    renewable_charge = arrays.renewable_charge.reshape(-1)
    grid_charge = arrays.grid_charge.reshape(-1)
    discharge = arrays.discharge.reshape(-1)
    grid_purchase = arrays.grid_purchase.reshape(-1)
    renewable_export = arrays.renewable_export.reshape(-1)
    curtailment = arrays.curtailment.reshape(-1)
    net = grid_purchase - renewable_export
    output = pd.DataFrame(
        {
            "Scheme": scheme,
            "Hour": frame["Hour"].astype(int),
            "Region": frame["Region"].astype(str),
            "TimeRole": frame["TimeRole"].astype(str),
            "Fixed_Facility_Load_MW": frame["Fixed_Facility_Load_MW"].to_numpy(dtype=float),
            "AvailableRenewable_MW": frame["AvailableRenewable_MW"].to_numpy(dtype=float),
            "RenewableDirectUse_MW": direct,
            "RenewableCharge_MW": renewable_charge,
            "GridCharge_MW": grid_charge,
            "ChargePower_MW": renewable_charge + grid_charge,
            "DischargePower_MW": discharge,
            "GridPurchase_MW": grid_purchase,
            "RenewableExport_MW": renewable_export,
            "RenewableCurtailment_MW": curtailment,
            "NetGridImport_MW": net,
            "SOC_MWh": arrays.soc[1:].reshape(-1),
            "OperatingCost_CNY": price * grid_purchase - sell_price * renewable_export,
            "CarbonEmission_tCO2": carbon * grid_purchase,
            "TerminalStateOnly": False,
        }
    )
    terminal = data.terminal.sort_values(["Hour", "Region"], kind="stable").copy().reset_index(drop=True)
    terminal_soc = arrays.soc[-1]
    if scheme == "BaselineReference":
        source_terminal = data.baseline_terminal.sort_values(
            "Region", kind="stable"
        )
        if len(source_terminal) != len(terminal):
            raise ValueError("附件基准终端状态与Q3终端区域数不一致")
        terminal_soc = source_terminal["SOC_MWh"].to_numpy(dtype=float)
    terminal_output = pd.DataFrame(
        {
            "Scheme": scheme,
            "Hour": terminal["Hour"].astype(int),
            "Region": terminal["Region"].astype(str),
            "TimeRole": "terminal",
            "Fixed_Facility_Load_MW": terminal["Fixed_Facility_Load_MW"].to_numpy(dtype=float),
            "AvailableRenewable_MW": terminal["AvailableRenewable_MW"].to_numpy(dtype=float),
            "RenewableDirectUse_MW": 0.0,
            "RenewableCharge_MW": 0.0,
            "GridCharge_MW": 0.0,
            "ChargePower_MW": 0.0,
            "DischargePower_MW": 0.0,
            "GridPurchase_MW": 0.0,
            "RenewableExport_MW": 0.0,
            "RenewableCurtailment_MW": 0.0,
            "NetGridImport_MW": 0.0,
            "SOC_MWh": terminal_soc,
            "OperatingCost_CNY": 0.0,
            "CarbonEmission_tCO2": 0.0,
            "TerminalStateOnly": True,
        }
    )
    return pd.concat([output, terminal_output], ignore_index=True)


def _audit_solution(data: Q3Data, model: LinearModel, solution: Solution) -> pd.DataFrame:
    arrays = _solution_arrays(model, solution)
    frame = data.operating.sort_values(["Hour", "Region"], kind="stable")
    time_count = len(data.hours)
    region_count = len(data.regions)
    fixed_load = frame["Fixed_Facility_Load_MW"].to_numpy(dtype=float).reshape(time_count, region_count)
    renewable = frame["AvailableRenewable_MW"].to_numpy(dtype=float).reshape(time_count, region_count)
    residuals: dict[str, float] = {
        "RenewableBalanceResidual_MW": float(
            np.max(np.abs(renewable - arrays.direct_renewable - arrays.renewable_charge - arrays.renewable_export - arrays.curtailment))
        ),
        "EnergyBalanceResidual_MW": float(
            np.max(np.abs(arrays.grid_purchase + arrays.direct_renewable + arrays.discharge - fixed_load - arrays.grid_charge))
        ),
        "SOCRecurrenceResidual_MWh": float(
            np.max(
                np.abs(
                    arrays.soc[1:]
                    - arrays.soc[:-1]
                    - model.data_arrays["ChargeEfficiency"] * (arrays.renewable_charge + arrays.grid_charge)
                    + arrays.discharge / model.data_arrays["DischargeEfficiency"]
                )
            )
        ),
        "InitialSOCViolation_MWh": float(
            np.max(np.abs(arrays.soc[0] - model.data_arrays["InitialSOC_MWh"]))
        ),
        "TerminalSOCDeficit_MWh": float(
            np.maximum(model.data_arrays["InitialSOC_MWh"] - arrays.soc[-1], 0.0).max()
        ),
        "SOCLowerViolation_MWh": float(
            np.maximum(model.data_arrays["MinSOC_MWh"] - arrays.soc, 0.0).max()
        ),
        "SOCUpperViolation_MWh": float(
            np.maximum(arrays.soc - model.data_arrays["StorageCapacity_MWh"], 0.0).max()
        ),
        "ChargePowerViolation_MW": float(
            np.maximum(
                arrays.renewable_charge + arrays.grid_charge
                - model.data_arrays["MaxChargePower_MW"],
                0.0,
            ).max()
        ),
        "DischargePowerViolation_MW": float(
            np.maximum(arrays.discharge - model.data_arrays["MaxDischargePower_MW"], 0.0).max()
        ),
        "ChargeDischargeOverlap_MW": float(
            np.minimum(arrays.renewable_charge + arrays.grid_charge, arrays.discharge).max()
        ),
        "GridImportViolation_MW": float(
            np.maximum(arrays.grid_purchase - model.data_arrays["MaxGridImport_MW"], 0.0).max()
        ),
        "SellLimitViolation_MW": float(
            np.maximum(arrays.renewable_export - model.data_arrays["SellLimit_MW"], 0.0).max()
        ),
        "GridExportViolation_MW": float(
            np.maximum(arrays.renewable_export - model.data_arrays["MaxGridExport_MW"], 0.0).max()
        ),
        "GridChargeSourceViolation_MW": float(
            np.maximum(arrays.grid_charge - arrays.grid_purchase, 0.0).max()
        ),
        "DirectRenewableLoadViolation_MW": float(
            np.maximum(arrays.direct_renewable - fixed_load, 0.0).max()
        ),
    }
    rows = [
        {
            "Check": check,
            "MaxViolation": value,
            "Status": "PASS" if value <= AUDIT_TOLERANCE else "FAIL",
        }
        for check, value in residuals.items()
    ]
    return pd.DataFrame(rows)


def _baseline_audit(data: Q3Data, arrays: DispatchArrays) -> pd.DataFrame:
    frame = data.operating.sort_values(["Hour", "Region"], kind="stable")
    time_count = len(data.hours)
    region_count = len(data.regions)
    fixed_load = frame["Fixed_Facility_Load_MW"].to_numpy(dtype=float).reshape(time_count, region_count)
    renewable = frame["AvailableRenewable_MW"].to_numpy(dtype=float).reshape(time_count, region_count)
    checks = {
        "RenewableBalanceResidual_MW": np.max(
            np.abs(renewable - arrays.direct_renewable - arrays.renewable_charge - arrays.renewable_export - arrays.curtailment)
        ),
        "EnergyBalanceResidual_MW": np.max(
            np.abs(arrays.grid_purchase + arrays.direct_renewable + arrays.discharge - fixed_load - arrays.grid_charge)
        ),
    }
    return pd.DataFrame(
        [
            {"Check": name, "MaxViolation": float(value), "Status": "PASS" if value <= AUDIT_TOLERANCE else "FAIL"}
            for name, value in checks.items()
        ]
    )


def _write_table(frame: pd.DataFrame, filename: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES_DIR / filename, index=False, encoding="utf-8-sig", float_format="%.15g")


def _write_outputs(
    data: Q3Data,
    model: LinearModel,
    balanced: Solution,
    anchors: dict[str, Solution],
    best: dict[str, float],
    reference: dict[str, float],
    solver_rows: list[dict[str, object]],
) -> None:
    balanced_arrays = _solution_arrays(model, balanced)
    baseline_arrays = _baseline_arrays(data)
    no_storage_arrays = _no_storage_arrays(data)

    balanced_metrics = _metric_values_from_arrays(data, balanced_arrays)
    baseline_metrics = _metric_values_from_arrays(data, baseline_arrays)
    no_storage_metrics = _metric_values_from_arrays(data, no_storage_arrays)
    anchor_rows: list[dict[str, object]] = []
    for objective_name, solution in anchors.items():
        metrics = _metric_values(model, solution.vector)
        anchor_rows.append(
            {
                "Solution": f"Anchor_{objective_name}",
                "AnchorObjective": objective_name,
                **metrics,
                "Status": solution.status,
                "Success": solution.success,
            }
        )
    _write_table(pd.DataFrame(anchor_rows), "q3_anchor_metrics.csv")

    summary_rows = []
    for solution_name, metrics in (
        ("Balanced", balanced_metrics),
        ("BaselineReference", baseline_metrics),
        ("NoStorage", no_storage_metrics),
    ):
        summary_rows.append(
            {
                "Solution": solution_name,
                "Cost": metrics["Cost"],
                "Carbon": metrics["Carbon"],
                "Peak": metrics["Peak"],
                "Ramp": metrics["Ramp"],
                "Throughput": metrics["Throughput"],
                "TerminalSOCTotal_MWh": float(balanced_arrays.soc[-1].sum())
                if solution_name == "Balanced"
                else float(data.baseline_terminal["SOC_MWh"].sum())
                if solution_name == "BaselineReference"
                else float(no_storage_arrays.soc[-1].sum()),
            }
        )
    for objective_name in OBJECTIVE_NAMES:
        summary_rows[0][f"{objective_name}Best"] = best[objective_name]
        summary_rows[0][f"{objective_name}Reference"] = reference[objective_name]
    _write_table(pd.DataFrame(summary_rows), "q3_objective_summary.csv")

    comparison_rows = []
    for metric in ("Cost", "Carbon", "Peak", "Ramp", "Throughput"):
        balanced_value = balanced_metrics[metric]
        for scheme, metrics in (
            ("BaselineReference", baseline_metrics),
            ("NoStorage", no_storage_metrics),
        ):
            reference_value = metrics[metric]
            difference = balanced_value - reference_value
            comparison_rows.append(
                {
                    "Metric": metric,
                    "ReferenceScheme": scheme,
                    "Balanced": balanced_value,
                    "Reference": reference_value,
                    "Difference": difference,
                    "RelativeDifference": difference / abs(reference_value)
                    if abs(reference_value) > FLOAT_EPS
                    else np.nan,
                }
            )
    _write_table(pd.DataFrame(comparison_rows), "q3_baseline_comparison.csv")

    _profile_frame(data, "Balanced", balanced_arrays).to_csv(
        TABLES_DIR / "q3_balanced_region_hour.csv", index=False, encoding="utf-8-sig", float_format="%.15g"
    )
    _profile_frame(data, "BaselineReference", baseline_arrays).to_csv(
        TABLES_DIR / "q3_baseline_region_hour.csv", index=False, encoding="utf-8-sig", float_format="%.15g"
    )
    _profile_frame(data, "NoStorage", no_storage_arrays).to_csv(
        TABLES_DIR / "q3_no_storage_region_hour.csv", index=False, encoding="utf-8-sig", float_format="%.15g"
    )

    regional = pd.concat(
        [
            _regional_metrics(data, "Balanced", balanced_arrays),
            _regional_metrics(data, "BaselineReference", baseline_arrays),
            _regional_metrics(data, "NoStorage", no_storage_arrays),
        ],
        ignore_index=True,
    )
    _write_table(regional, "q3_regional_metrics.csv")

    audit = _audit_solution(data, model, balanced)
    _write_table(audit, "q3_constraint_audit.csv")
    _write_table(_baseline_audit(data, baseline_arrays), "q3_baseline_audit.csv")
    _write_table(pd.DataFrame(solver_rows), "q3_solver_log.csv")

    configuration = pd.DataFrame(
        [
            {
                "StartHour": data.hours[0],
                "OperationalEndHour": data.hours[-1],
                "TerminalHour": int(data.terminal["Hour"].iloc[0]),
                "OperationalHourCount": len(data.hours),
                "RegionCount": len(data.regions),
                "VariableCount": model.index.count,
                "BinaryVariableCount": int(model.integrality.sum()),
                "BaseConstraintCount": model.matrix.shape[0],
                "AuditTolerance": AUDIT_TOLERANCE,
                "BalanceTolerance": BALANCE_TOLERANCE,
                "Note": "2406小时只用于终端SOC状态结算；正式多目标方案为三级均衡择优结果。",
            }
        ]
    )
    _write_table(configuration, "q3_model_configuration.csv")
    logging.info("结果写入完成：%s。", TABLES_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="问题3储能时移多目标能源协同优化模型")
    parser.add_argument(
        "--max-hour",
        type=int,
        default=FULL_OPERATIONAL_END_HOUR,
        help="运行时段最后一个小时，默认2405；仅用于烟测时可设为较小值",
    )
    parser.add_argument(
        "--mip-rel-gap",
        type=float,
        default=1e-5,
        help="MILP相对最优间隙，默认1e-5；不设置时间上限",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=DEFAULT_SOLVER_PROGRESS_INTERVAL,
        help="MILP求解心跳日志间隔（秒），默认30；设为更小值可获得更密集进度",
    )
    parser.add_argument(
        "--solver-disp",
        action="store_true",
        help="同时打开HiGHS原生求解器输出；通常保留默认心跳日志即可",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="控制台和文件日志级别，默认INFO",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.max_hour <= FULL_OPERATIONAL_END_HOUR:
        raise SystemExit("--max-hour必须位于0--2405")
    if not 0 <= args.mip_rel_gap < 1:
        raise SystemExit("--mip-rel-gap必须位于[0,1)")
    if args.progress_interval <= 0:
        raise SystemExit("--progress-interval必须为正数")

    _configure_logging(args.log_level)
    started = time.perf_counter()
    logging.info(
        "问题3模型启动：运行时段0--%d，终端小时%d，MILP相对间隙=%g，心跳间隔=%.1fs，原生输出=%s，无时间上限。",
        args.max_hour,
        args.max_hour + 1,
        args.mip_rel_gap,
        args.progress_interval,
        args.solver_disp,
    )
    data = _load_data(args.max_hour)
    model = _build_linear_model(data)
    balanced, anchors, best, reference, solver_rows = _solve_multiobjective(
        model,
        args.mip_rel_gap,
        progress_interval=args.progress_interval,
        solver_disp=args.solver_disp,
    )
    _write_outputs(data, model, balanced, anchors, best, reference, solver_rows)
    audit = pd.read_csv(TABLES_DIR / "q3_constraint_audit.csv", encoding="utf-8-sig")
    failed_audits = audit.loc[audit["Status"] != "PASS"]
    if not failed_audits.empty:
        raise RuntimeError(f"Q3均衡方案约束审计未通过：{failed_audits.to_dict(orient='records')}")
    logging.info(
        "问题3模型完成：Cost=%.8g，Carbon=%.8g，Peak=%.8g，Ramp=%.8g，总耗时%.1fs。",
        _metric_values(model, balanced.vector)["Cost"],
        _metric_values(model, balanced.vector)["Carbon"],
        _metric_values(model, balanced.vector)["Peak"],
        _metric_values(model, balanced.vector)["Ramp"],
        time.perf_counter() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
