# Mesopotamia Project - SDR Underground Detection System Development

## Plan
- [X] 001 analyze_github_repository()
- [ ] 002 setup_development_environment()
- [ ] 003 develop_core_modules()
- [ ] 004 implement_signal_processing_algorithms()
- [ ] 005 create_visualization_components()
- [ ] 006 integrate_ai_detection_models()
- [ ] 007 test_and_optimize_code()
- [ ] 008 prepare_final_deliverables()

## Step 002: Setup Development Environment
- [X] Create `todo.md` file.
- [X] Identify Python dependencies from existing code.
- [X] Create `requirements.txt` file.
- [X] Install dependencies in the sandbox environment (using pip).
- [X] Provide instructions for setting up the environment on Windows using Anaconda/PyCharm.

## Step 003: Develop Core Modules
- [X] Refactor/rename `ai_classifier.py` to `calibration.py` (Removed duplicate `ai_classifier.py`).
- [X] Create a new `ai_classifier.py` for actual AI classification logic.
- [X] Update `main.py` to reflect module changes and dependencies.
- [X] Develop `config.py` or enhance `config_manager.py` for centralized configuration (enums, constants, hardware specs).
- [X] Populate `system_config.json` with initial settings.
- [ ] Implement `mobile_sync.py` structure (placeholder for now, actual implementation might be complex).
- [X] Refine `logger.py` or log## Step 004: Implement Signal Processing Algorithms
- [X] Enhance `sdr_receiver.py`:
    - [X] Implement dual-antenna noise suppression logic (using Dipole + UWB).
    - [X] Implement multi-frequency scanning logic (200MHz-3GHz range selection).
    - [X] Refine adaptive gain control.
    - [X] Add TX functionality placeholders (if needed, SoapySDR supports TX).
- [X] Enhance `calibration.py`:
    - [X] Improve ground type identification based on multi-frequency sweeps.
    - [X] Refine feature extraction for different soil types.
    - [X] Implement initial calibration sequence using Dipole antenna (Rx2).
    - [X] Implement continuous background noise monitoring (using UWB when idle or Dipole).
- [X] Create `preprocessing.py`:
    - [X] Implement advanced filtering (beyond basic bandpass/notch) - Wavelet Denoising.
    - [X] Implement echo-time analysis for depth estimation.
    - [X] Implement phase/amplitude difference analysis for density/depth.
    - [X] Implement algorithms to differentiate natural features (cracks, roots) from targets (Placeholder).
    - [X] Implement algorithms for handling challenging terrains (rocks, mineralized soil, metal trash) (Placeholder).## Step 005: Create Visualization Components
- [X] Enhance `depth_3d_view.py`:
    - [X] Implement real-time 3D model updates (using data from `preprocessing.py`).
    - [X] Implement color coding as specified (soil, stone, void, metal) using `config.py` maps.
    - [X] Implement depth representation (opacity/brightness).
    - [X] Ensure compatibility with Plotly/PyVista.
- [X] Enhance `radar_ui.py`:
    - [X] Implement the two-panel layout (Left: 3D Map, Right: 2D Density/Signal Map).
    - [X] Implement 2D signal intensity map (raw data view) (Placeholder using Matplotlib).
    - [X] Implement real-time confidence score display panel (Target List Tab).
    - [X] Add UI controls for frequency selection, calibration start, etc.
    - [ ] Ensure mobile compatibility aspects (responsive design if possible within desktop UI framework, or note limitations) - Not feasible with PyQt.
    - [X] Integrate UI updates with other modules via ZMQ.

## Step 006: Integrate AI Detection Models
- [X] Implement `ai_classifier.py`:
    - [X] Define input/output structure for AI model.
    - [X] Implement pattern recognition logic (Rule-based implemented, placeholder for deep learning model).
    - [X] Implement void-metal-stone differentiation logic (Rule-based).
    - [X] Implement material and shape estimation logic (basic, rule-based).
    - [X] Implement false positive reduction logic (hot rocks, roots, moisture, metal trash, disturbed soil) (Rule-based).
    - [X] Calculate and output confidence scores for detections.
- [X] Integrate AI results with `visualization.py` (`depth_3d_view.py`, `radar_ui.py`) (AI output format defined, UI consumes it).
- [ ] Note: Actual deep learning model training is outside the scope of code generation but structure will be provided.

## Step 007: Test and Optimize Code
- [X] Develop `test_system.py` with unit/integration tests for key modules.
- [X] Refine `monitoring.py` for resource usage tracking.
- [X] Optimize signal processing and AI algorithms for real-time performance (Reviewed `preprocessing.py` and `ai_classifier.py`).
- [ ] Perform code review and cleanup.

## Step 008: Prepare Final Deliverables
- [ ] Consolidate all code files into the `/home/ubuntu/mesopotamia_project/Toprakperest` directory.
- [ ] Ensure all dependencies are listed in `requirements.txt`.
- [ ] Write a `README.md` with setup instructions (including Windows/Anaconda/PyCharm), usage guide, and architecture overview.
- [ ] Package the project files into a zip archive.
- [ ] Send the final code and documentation to the user.
