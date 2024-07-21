# XIRR Calculator App

## Overview

This XIRR (Extended Internal Rate of Return) Calculator App is designed to help you calculate the XIRR 
for a series of cash flows. By leveraging PySpark, this application aims to provide faster calculations, 
especially for large datasets. This is helpful for tracking performance of your investments.

## Features

- **XIRR Calculation:** Accurately calculate the XIRR for your investments.
- **PySpark Integration:** Utilize PySpark for handling large datasets and faster computation.

## Work in Progress

This project is currently still work in progress. Please note that big data calculations maybe unstable with the current approach. 

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/nnn007/xirr-calculator.git
    cd xirr-calculator
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Application:**
    ```bash
    python app_quantized.py
    ```

2. **Input Cash Flows:**
    - Follow the prompts to input your series of cash flows (dates and amounts).

3. **View XIRR:**
    - The application will compute and display the XIRR for the provided cash flows.

## Example

- **Cash Flow Input:**
    ```plaintext
    Date,Amount
    2000-01-04 15:40:16.201620162,-8880.792198175182
    2000-01-08 07:20:32.403240324,-1293.0938906917236
    2000-01-08 07:20:32.403240324,6110.691696628242
    2000-01-08 07:20:32.403240324,-1676.5432965069376
    .....
    ```

## Contributing

Contributions are welcomed for this project! Please follow these steps:

1. **Fork the Repository**
2. **Create a New Branch**
    ```bash
    git checkout -b feature-branch
    ```
3. **Commit Your Changes**
    ```bash
    git commit -m "Add new feature"
    ```
4. **Push to the Branch**
    ```bash
    git push origin feature-branch
    ```

5. **Create a Pull Request**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please open an issue or reach out to us at [nayan.nilesh@gmail.com](mailto:nayan.nilesh@gmail.com).
