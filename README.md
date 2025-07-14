# HeatPump-FlexModeling-Toolkit

This toolkit provides a flexible framework for grey-box modeling, simulation, and parameter optimization of heat pump and building thermal systems. It supports multiple model structures (1R1C, 2R2C, 3R2C, 4R3C) and includes data preprocessing, training, validation, and visualization.

## Features

- Data loading and preprocessing from Excel files
- Support for multiple thermal model structures
- Automated training with multiple trials and objective evaluation
- Visualization of temperature, thermal power, and parameter sensitivity
- Model validation and residual analysis

## Folder Structure

- `Tutorial/`: Example Jupyter notebooks for case studies and model runs
- `src/`: Source code for models, simulation, plotting, and training
- `Journal Paper Implementation - Tuning Performance of Grey-Box Models for Thermal Building Applications/`: Additional notebooks for performance evaluation

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- FinalToolModels (custom module)
- ipopt (solver)

## Usage

1. Place your dataset in the `Data` directory.
2. Open and run the Jupyter notebooks in `Tutorial/` (e.g., `Case Study 1.ipynb`) to execute the workflow.
3. Adjust model bounds and parameters as needed for your case study.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
