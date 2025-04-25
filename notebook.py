

import marimo

__generated_with = "0.13.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        Webscraping
        ===
        MAIC - Spring, Week 3<br>
        ```
          _____________
         /0   /     \  \
        /  \ M A I C/  /\
        \ / *      /  / /
         \___\____/  @ /
                  \_/_/
        ```
        (Rosie is not needed!)

        **Prereqs [IMPORTANT!]:**
        - Install Requests in the "Manage Packages" tab (remove the original `requests` package first).
        - Install Pydantic-AI in the "Manage Packages" tab.

        <span style="color:#ff5555;font-weight:bold;font-size:1.5rem;">
            STOP
        </span>

        ... or keep going if you want to work ahead.

        ---
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **What is webscraping?**

        Are you in need of data? Maybe you want to analyze some data for insights. Or maybe you just want to train a model. In any case, you may be able to get the data you need via webscraping!

        Webscraping is the process of *automatically* extracting data from websites. You can manually extract website data on your browser via "inspect," but automating this process is ideal if you need anything more than a few samples.

        - Go to any website (for instance, the [MAIC](https://msoe-maic.com/) site).
        - Right-click anywhere on the page. Select the "inspect" option or something labeled similarly. This is usually at the bottom of the pop-up menu.
        - Note the window that opened. It contains the raw HTML (and possibly JS/CSS) site data. This is what we want to scrape automatically.
        - Use the element selector at the top left of the inspect window to see the HTML of specific elements.

        ---

        <span style="color:#55ff55;font-weight:bold;font-size:1.5rem;">
            GO
        </span>

        **That's cool. How can I scrape automatically?**

        Let's try scraping the MAIC leaderboard!

        Basic scraping only needs the `requests` library.
        """
    )
    return


@app.cell
def _():
    import requests

    URL = 'https://msoe-maic.com'

    html = requests.get(URL).text # Make a request to the URL and get the HTML via `.text`
    print(html[:500]) # Print some of the resulting HTML
    return html, requests


@app.cell
def _(mo):
    mo.md(
        r"""
        This html now contains the leaderboard for us to extract. But how do we extract it?

        One easy way is to *inspect* the page on your browser, and to see if the HTML can easily identify the leaderboard. It seems that the leaderboard element is in the "leaderboard-table" class:

        ```html
        <table border="1" class="dataframe leaderboard-table" id="df_data">
            ...
        </table>
        ```

        We could try looking for "leaderboad-table" in the html string, but there's a better way. `Beautifulsoup` is a Python library that makes parsing HTML easy.
        """
    )
    return


@app.cell
def _(html):
    from bs4 import BeautifulSoup # We can now use BeautifulSoup to parse the HTML

    soup = BeautifulSoup(html, 'html.parser') 
    print(soup.prettify()[:500]) # print it as before, but now it's prettified
    return (soup,)


@app.cell
def _(mo):
    mo.md(r"""Now we can use BeautifulSoup to find the "leaderboard-table" element.""")
    return


@app.cell
def _(soup):
    # find the table element with class "leaderboard-table"

    leaderboard_table = soup.find('table', {'class': 'leaderboard-table'})

    print(leaderboard_table.prettify()[:500]) # print the first 500 characters of the table
    return (leaderboard_table,)


@app.cell
def _(mo):
    mo.md(r"""Not only can Beautifulsoup find the element, it also allows us to easily extract the data.""")
    return


@app.cell
def _(leaderboard_table):
    # Extract table data into a list of dictionaries

    rows = leaderboard_table.find_all('tr') # Find all rows in the table
    header = [cell.text for cell in rows[0].find_all('th')] # Get the header row
    data = [
        {header[i]: cell.text for i, cell in enumerate(row.find_all('td'))} # Create a dictionary for each row using the header to name the keys
        for row in rows[1:]
    ]

    data
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Pretty neat, right?

        <span style="color:#ff5555;font-weight:bold;font-size:1.5rem;">
            STOP
        </span>

        ... or keep going if you want to work ahead.

        ---

        **It's not always this easy.**

        Some pages dynamically generate content using Javascript. This is a problem for us because the `requests` library cannot run Javascript code. Let's try to scrape content from a page that uses a lot of Javascript.

        - Go to [the MAIC research groups page](https://msoe-maic.com/library?nav=Research).
        - Use the element selector to select a group's section.
        - Note the id of the element.

        For instance, the page has this div with an id of `agent-simulation-experts`.

        ```html
        <div class="MuiPaper-root MuiPaper-elevation MuiPaper-rounded MuiPaper-elevation1 MuiCard-root modal css-1kil0ip" id="agent-simulation-experts">
            ...
        </div>
        ```

        It's important to note, however, that this element was generated with Javascript. So what happens if we try scraping this element with `requests`?

        <span style="color:#55ff55;font-weight:bold;font-size:1.5rem;">
            GO
        </span>
        """
    )
    return


@app.cell
def _(requests):
    URL_1 = 'https://msoe-maic.com/library?nav=Research'
    html_1 = requests.get(URL_1).text
    print(html_1)
    return (html_1,)


@app.cell
def _(html_1):
    print('agent-simulation-experts' in html_1)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <span style="color:#5555ff;font-weight:bold;">
            Try this yourself.
        </span>

        Go to some websites and see what HTML you can scrape with `requests`. See if anything in the browser inspection tool appears in the `html` variable. You may find that a majority of websites aren't easily scrapable.

        Some sites to try:
        - https://www.youtube.com/
        - https://www.google.com/search?q=your+search+here
        - https://www.reddit.com/
        - https://stackoverflow.com/questions
        - https://github.com/torvalds

        <span style="color:#ff5555;font-weight:bold;font-size:1.5rem;">
            STOP
        </span>

        ... or keep going if you want to work ahead.

        ---

        **Q: So how do we scrape pages that use Javascript?**

        A: Use Selenium.

        Selenium is a headless browser that can execute page Javascript.

        the `requests` library cannot run Javascript, so any page content generated by said Javascript is impossible to scrape with `requests` alone. Luckily, browsers are *made* to run Javascript. Selinum runs javascript like a regular browser (and it even uses a regular browser such as Chrome under the hood), but it functions without a UI so you can interact with pages programatically


        <span style="color:#55ff55;font-weight:bold;font-size:1.5rem;">
            GO
        </span>

        We'll wrap `selenium` in a function call to make it work similarly to `requests`. Feel free to read the function comments if you want to dive deeper into `selenium`.
        """
    )
    return


@app.cell
def _(BeautifulSoup_1):
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.edge.service import Service as EdgeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.microsoft import EdgeChromiumDriverManager
    from bs4 import BeautifulSoup
    import time

    def setup_chrome():
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        print('Opening Chrome Webdriver')
        return webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

    def setup_edge():
        options = EdgeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        print('Opening Edge Webdriver')
        return webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()), options=options)
    driver = None
    try:
        driver = setup_chrome()
    except Exception as e:
        print(f'Chrome failed')
        print('Falling back to Edge...')
        driver = setup_edge()

    def get_page_content(url):
        """
        Opens a URL using Selenium and retrieves the page contents.
        Tries Chrome first, falls back to Edge if Chrome fails.

        Args:
            url (str): The URL to open

        Returns:
            tuple: (raw_html, parsed_text) where raw_html is the page source and 
                   parsed_text is the cleaned text content
        """
        print('Scrape')
        try:
            driver.get(url)
            time.sleep(2)
            page_content = driver.page_source
            soup = BeautifulSoup_1(page_content, 'html.parser')
            for script in soup(['script', 'style']):
                script.decompose()
            code_blocks = soup.find_all(['pre', 'code'])
            for block in code_blocks:
                if block.string:
                    block.string = f'```{block.string}```'
                else:
                    block.string = f'```{block.get_text()}```'
            text = soup.get_text().replace('```Copy', '```')
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split('  '))
            text = '\n'.join((chunk for chunk in chunks if chunk))
            return (page_content, text)
        except Exception as e:
            print(f'An error occurred: {str(e)}')
            return (None, None)
    return driver, get_page_content


@app.cell
def _(get_page_content):
    URL_2 = 'https://msoe-maic.com/library?nav=Research'
    (html_2, _) = get_page_content(URL_2)
    return (html_2,)


@app.cell
def _(BeautifulSoup_1, html_2):
    soup_1 = BeautifulSoup_1(html_2, 'html.parser')
    agent_sim_div = soup_1.find('div', {'id': 'agent-simulation-experts'})
    modal_header = agent_sim_div.find(class_='modal-header')
    print(modal_header.get_text().strip() if modal_header else 'No modal-header found')
    print(agent_sim_div)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <span style="color:#ff5555;font-weight:bold;font-size:1.5rem;">
            STOP
        </span>

        ... or keep going if you want to work ahead.

        ---

        **Using LLMs to summarize scraped data.**

        If you're scraping unstructured data, then LLMs are a must. Although there is structure in the HTML elements, it can often be easier to ask an LLM to structure the output for you.

        Let's structure the output of a page listing refurbished iPhones for sale.

        You will need Gemini API keys to run this example. [Link to Gemini API](https://aistudio.google.com/)

        <span style="color:#55ff55;font-weight:bold;font-size:1.5rem;">
            GO
        </span>

        This example will:
        - Use Selenium to scrape for refurbished iPhones.
        - Use an LLM to summarize the results into a structured format.
        """
    )
    return


@app.cell
def _():
    import os
    from pydantic_ai import Agent
    from typing import List
    from pydantic import BaseModel, Field
    return Agent, BaseModel, Field, List, os


@app.cell
def _(os):
    URL_3 = 'https://www.apple.com/shop/refurbished/iphone/iphone-14-pro'
    os.environ['GEMINI_API_KEY'] = ...
    return (URL_3,)


@app.cell
def _(BaseModel, Field, List):
    # This is where you can specify the output structure

    class ProductResult(BaseModel):  
        model: str = Field(description='The model of the product')
        description: str = Field(description='The description of the product')
        cost: int = Field(description="The cost of the product")
        isp: str = Field(description="The internet service provider")
        color: str = Field(description="The color of the product")
        refurbished: bool = Field(description="Whether the product is refurbished")

    # We are storing a list of ProductResults in the final output
    class RequestResults(BaseModel):
        products: List[ProductResult] = Field(description='The list of product results')
    return (RequestResults,)


@app.cell
def _(Agent, RequestResults):
    # "Here's where the AI comes in" - Andrej Karpathy
    agent = Agent( # Create an agent that will structure the output
        'google-gla:gemini-1.5-flash',
        result_type=RequestResults,
        system_prompt='Be concise, reply with one sentence.',  
    )

    # Agent system prompt - tell it what to do
    @agent.system_prompt  
    async def add_customer_name(ctx) -> str:
        return f"Your goal is to extract product information from web scraped pages and format it to a structured response."
    return (agent,)


@app.cell
def _(URL_3, get_page_content):
    (_, text) = get_page_content(URL_3)
    return (text,)


@app.cell
async def _(agent, text):
    result = await agent.run(text)
    return (result,)


@app.cell
def _(result):
    import pandas as pd
    from IPython.display import display

    # Structure the output into a DataFrame to see our results
    item_dicts = [item.model_dump() for item in result.data.products]
    df = pd.DataFrame(item_dicts)
    display(df.head(20))
    return


@app.cell
def _(driver):
    driver.quit() # Always remember to close the webdriver when you're done with it
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
