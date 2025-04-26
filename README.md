# Workshop Template
Developed by: Adam Haile - 2025

## Requirements
`pipenv`

## How to use
1. Clone the repository to your local machine.
2. Run `pipenv install` to setup the marimo environment
3. Run `pipenv run marimo` to start the marimo web server
4. Create a new notebook. Title it something along the lines of `<workshop name>.py`
    > Note: ensure your files start with `import marimo as mo` so rendering works correctly.
5. Write your workshop, ensure it all runs.
    > Note: install packages from the "Manage Packages" option on the left side
6. Once your workshop is ready, edit the `export.sh` file and add the name of your workshop to the list at the top
7. Run `pipenv run export` to build the WASM code.
8. Verify it works by running `pipenv run host` and going to https://localhost:8000
9. Commit/push you changes

Verify the workshop works by going to the link with your workshop. Also make sure you copy this to put on the main MAIC website.
