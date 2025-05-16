# Stage Summary: ConsumptionSavingsIID

## Stage Model
parameters:
  - beta: float
  - r: float
  - sigma: float
  - gamma: int
  - sigma_y: float
    ... (1 more non-empty items)
settings:
  - tol: float
  - max_iter: int
  - N_shock: int
  - n_grid_points: int
  - a_min: str
    ... (11 more non-empty items)
math:
  - functions: dict
  - constraints: dict
  - shocks: dict
  - state: dict
num:
  - functions: dict
  - constraints: dict
  - state_space: dict
  - shocks: dict
  - arvl: dict
    ... (2 more non-empty items)

## Perches
### Perch: arvl
- sol: NoneType
- dist: NoneType
- model:
    parameters:
      - beta: float
      - r: float
      - sigma: float
      - gamma: int
      - sigma_y: float
        ... (1 more non-empty items)
    settings:
      - tol: float
      - max_iter: int
      - N_shock: int
      - n_grid_points: int
      - a_min: str
        ... (11 more non-empty items)
    math:
      - state: dict
    num:
      - state_space: dict
      - arvl: dict

### Perch: dcsn
- sol: NoneType
- dist: NoneType
- model:
    parameters:
      - beta: float
      - r: float
      - sigma: float
      - gamma: int
      - sigma_y: float
        ... (1 more non-empty items)
    settings:
      - tol: float
      - max_iter: int
      - N_shock: int
      - n_grid_points: int
      - a_min: str
        ... (11 more non-empty items)
    math:
      - state: dict
    num:
      - state_space: dict
      - dcsn: dict

### Perch: cntn
- sol: dict
- dist: NoneType
- model:
    parameters:
      - beta: float
      - r: float
      - sigma: float
      - gamma: int
      - sigma_y: float
        ... (1 more non-empty items)
    settings:
      - tol: float
      - max_iter: int
      - N_shock: int
      - n_grid_points: int
      - a_min: str
        ... (11 more non-empty items)
    math:
      - state: dict
    num:
      - state_space: dict
      - cntn: dict

## Movers
### Mover: arvl_to_dcsn (forward)
- source: arvl
- target: dcsn
- model:
    parameters:
      - beta: float
      - r: float
      - sigma: float
      - gamma: int
      - sigma_y: float
        ... (1 more non-empty items)
    settings:
      - tol: float
      - max_iter: int
      - N_shock: int
      - n_grid_points: int
      - a_min: str
        ... (11 more non-empty items)
    math:
      - functions: dict
      - constraints: dict
      - shocks: dict
      - state: dict
    num:
      - functions: dict
      - constraints: dict
      - state_space: dict
      - shocks: dict
      - arvl: dict
        ... (1 more non-empty items)

### Mover: dcsn_to_cntn (forward)
- source: dcsn
- target: cntn
- model:
    parameters:
      - beta: float
      - r: float
      - sigma: float
      - gamma: int
      - sigma_y: float
        ... (1 more non-empty items)
    settings:
      - tol: float
      - max_iter: int
      - N_shock: int
      - n_grid_points: int
      - a_min: str
        ... (11 more non-empty items)
    math:
      - functions: dict
      - constraints: dict
      - state: dict
    num:
      - functions: dict
      - constraints: dict
      - state_space: dict
      - dcsn: dict
      - cntn: dict

### Mover: cntn_to_arvl (forward)
- source: cntn
- target: arvl
- model: None

### Mover: dcsn_to_arvl (backward)
- source: dcsn
- target: arvl
- model:
    parameters:
      - beta: float
      - r: float
      - sigma: float
      - gamma: int
      - sigma_y: float
        ... (1 more non-empty items)
    settings:
      - tol: float
      - max_iter: int
      - N_shock: int
      - n_grid_points: int
      - a_min: str
        ... (11 more non-empty items)
    math:
      - functions: dict
      - constraints: dict
      - shocks: dict
      - state: dict
    num:
      - functions: dict
      - constraints: dict
      - state_space: dict
      - shocks: dict
      - dcsn: dict
        ... (1 more non-empty items)

### Mover: cntn_to_dcsn (backward)
- source: cntn
- target: dcsn
- model:
    parameters:
      - beta: float
      - r: float
      - sigma: float
      - gamma: int
      - sigma_y: float
        ... (1 more non-empty items)
    settings:
      - tol: float
      - max_iter: int
      - N_shock: int
      - n_grid_points: int
      - a_min: str
        ... (11 more non-empty items)
    math:
      - functions: dict
      - constraints: dict
      - state: dict
    num:
      - functions: dict
      - constraints: dict
      - state_space: dict
      - cntn: dict
      - dcsn: dict
