# Workshop Template
Developed by: Adam Haile - 2025

## Requirements
`pipenv`

## How to use
1. Create a fork of the workshop-template.
2. Name the fork the name of the workshop you are building, along with update the description.
3. Clone the forked repository to your local machine.
4. Run `pipenv install` to setup the marimo environment
5. Run `pipenv run marimo` to start the marimo web server
6. Either edit the original `notebook.py` or delete it and start fresh with a new `notebook.py`
    > IMPORTANT: The file **must** be named `notebook.py`. If you name it something else, you will also have to update the Pipfile export. If you do not know how to do this, just name it `notebook.py`
    > Additionally, ensure your files start with `import marimo as mo` so rendering works correctly.
7. Write your workshop, ensure it all runs. (Note, install packages from the "Manage Packages" option on the left side)
8. Once your workshop is ready, run `pipenv run export` to build the WASM code.
9. Commit/push you changes
10. On the Github repo page, go to the "Settings" tab
11. Go to "Pages"
12. On the dropdown next to "Branch", choose "main". Then press "Save"
13. After a few moments, refresh the page. You will eventually see the Github page link appear.

Verify the workshop works by going to the link. Also make sure you copy this to put on the main MAIC website.
