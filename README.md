# PPTD: A Path Profiling-based Threat Detection Method Towards Deployed Smart Contracts

PPTD is a path profiling-based smart contract threat detection method that performs comprehensive path profiling to detect potential vulnerabilities and monitor execution traces.

## Overview

PPTD leverages control flow graph (CFG) analysis and path profiling algorithms to instrument smart contracts with monitoring code that tracks execution paths at runtime. The framework intelligently selects between two path profiling algorithms based on the contract structure:

- **PAP (Profiling ALL Paths)**: For contracts containing loops or cyclic control flow structures
- **EPP (Efficient Path Profiling)**: For acyclic contracts without loops

## Architecture

The PPTD framework consists of several key components:

### Current Modules

- **`CFG/`**: Control flow graph analysis and construction
  - `cfg_analyzer.py`: CFG builder and analyzer for contract functions
  - `contract_graph.py`: Graph data structures and operations for CFG representation
- **`Encoding/`**: Path profiling algorithms and instrumentation
  - `path_profiler.py`: EPP and PAP algorithm implementations with unified interface
  - `Instrumentation/`: Code instrumentation utilities
    - `instrument.py`: Main code instrumenter with EPP and PAP support
    - `ins_bytecode.py`: Bytecode-level instrumentation utilities for EVM operations
    - `ins_tex.py`: $tex$ variable instrumentation for reentrancy protection
- **`Preprocessing/`**: Contract preprocessing and transformation utilities
  - `preprocess.py`: Smart contract preprocessing

*Other modules will be released soon.*

## Usage

### Vulnerabilities with Automatic Detection and Prevention

These vulnerabilities trigger abnormal control flows, which PPTD identifies via legal path validation and auto-prevents without manual intervention.

| Vulnerability Type | Abbreviation | Description |
|--------------------|--------------|-------------|
| Reentrancy | RE | Attackers exploit external calls to re-enter contract functions (e.g., stealing funds via fallback functions) |
| Unsafe Delegatecall | UD | `delegatecall()` modifies the caller’s state using the callee’s logic, leading to unintended variable overwrites when the callee’s data structure changes. |
| Suicidal Contract | SC | Contracts use `selfdestruct()` without setting function visibility, allowing attackers to call the function, destroy the contract, and steal funds (e.g., Zapit contract). |
| Loop Vulnerability | LV | Attackers expand loop iterations (e.g., creating numerous investor accounts) to exceed the block gas limit, rendering functions (e.g., `distribute()`) inoperable. |
| Assertion Failure | AF | `assert()` (which consumes all remaining gas) fails due to contract errors or incorrect usage; PPTD replaces `assert()` with `require()` to save gas. |
| Integer Bug | IB | Integer overflow/underflow occurs when contracts use unvalidated user input for calculations, exceeding EVM’s fixed-size integer limits. |
| Ether Leak | EL | Non-owner users obtain contract Ether by sending transactions, triggering a special execution path for unauthorized transfers. |
| Unhandled Exception | UE | The `CALL` instruction returns `false` (indicating an exception) but lacks error handling, failing to roll back the entire program . |
| tx.Origin Use | TU | Using `tx.origin` (a global variable returning the original transaction sender) for authentication enables phishing-style attacks via fallback calls. |
| Arbitrary Call | AC | Unauthorized entities (users/other contracts) initiate unintended function calls due to insufficient access controls or input validation. |

### Vulnerabilities with Detection Only (No Automatic Prevention)

These vulnerabilities primarily affect data flow (not control flow) or require business logic adjustments. PPTD detects them via the fuzzer but requires manual repair by contract creators.

| Vulnerability Type | Abbreviation | Description |
|--------------------|--------------|-------------|
| Unexpected Revert | UR | Results of external calls incorrectly trigger the `REVERT` instruction branch, disrupting contract execution. |
| Block State Dependency | BS | Block state variables (e.g., `BLOCKHASH`, `TIMESTAMP`, `NUMBER`, `DIFFICULTY`) flow into operands of sensitive opcodes (e.g., `CALL`, `JUMPI`), enabling manipulation. |
| Transaction Order Dependency | TO | The first transaction sender writes to a storage location that the second sender reads, allowing attackers to exploit transaction ordering (e.g., front-running). |
| Arbitrary Write | AW | Unauthorized users modify arbitrary contract storage slots due to insufficient validation of write operations. |
| Freezing Ether | FE | Contract design traps Ether (e.g., missing withdrawal functions), making funds inaccessible to users. |

### Hybrid Instrumentation Strategy

PPTD implements an intelligent instrumentation strategy:

#### For Loop Contracts

- **Primary**: PAP instrumentation for handling cyclic execution paths
- **Secondary**: EPP instrumentation for acyclic portions
- **Benefit**: Complete coverage of both loop and non-loop execution scenarios

#### For Non-Loop Contracts

- **Single**: EPP instrumentation only
- **Benefit**: Optimized performance without unnecessary overhead

## Research Applications

PPTD is designed for research in:

- Post-deployment protection
- Smart contract security analysis
- Dynamic execution monitoring
- Path-based vulnerability detection
- Runtime behavior analysis
- Gas optimization studies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, issues, or collaboration opportunities, please contact:

- Email: [dingk986@163.com]

## Acknowledgments

The design inspiration for building PPTD comes from [ContractGuard](https://github.com/contractguard/experiments) and [EtherProv](https://github.com/shomzy/EtherProv). We are very grateful for the related works [Slither](https://github.com/crytic/slither) and [SMARTIAN](https://github.com/SoftSec-KAIST/Smartian). They laid the foundation for our framework and code. Thank you for their excellent work.

