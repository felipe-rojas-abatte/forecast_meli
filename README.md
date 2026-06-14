# Mercado Libre Technical Test

Mercado Libre operates in multiple countries in Latin America, serving from various warehouses and fulfillment centers. Currently, we process more than 50 orders per second. To meet our customers' expectations, we need to have the most accurate demand forecast possible to have the right stock in each city and reach our buyers as quickly as possible.

In this technical test, you must develop a solution to generate the requested estimates. Since we mostly work with Python, we ask you to do the same for this test; however, you can also use other technologies if needed along with any library you deem appropriate.

## Data
To solve this test, two datasets have been included. One contains product sales information, and the other contains geographic information regarding the place from where the sale was made.

Each product is served from a specific city if the associated Zipcode is within the range defined in the `geo.csv` file.

Example:
Product with ID `d8371be8-234e-3289-8j64-10e658ce3002` is associated with Zipcode `44639999`. From the `geo` file, it is seen that city `B1` is associated with the range: 44600000-44640000. Therefore, this product is served from this city.

### File Descriptions
***product_sales.csv***
- product_id: Product ID.
- country: Country where the sale occurred.
- date: Date the sale was made.
- zipcode: Zipcode from which the product is served.
- sales: Number of product sales for the date and country.

***geo.csv:***
- country: Country associated with the order.
- s_zipcode: Starting Zipcode for the range.
- e_zipcode: Ending Zipcode for the range.
- city: City from which a product is served.

*All information has been anonymized*

## Objective

The objective is to generate a weekly demand forecast at the `product_id` level for each city for the next 3 days from the last sale date in the `product_sales` file. For example, if the last sales day was August 8, 2024, the forecast should be delivered for August 9, 10, and 11, 2024. The forecast for August 9 equals the projected sales from August 9 to 15 (1 week) and so on.

For this, a file (`submission.csv`) with the expected format has been attached.

Since we are interested in seeing how you approach the problem, we ask you to include the code used to generate the forecast, along with any analysis you have performed and consider relevant. The code should be documented, and its results should be easily replicable, ideally self-contained, and clear regarding the decisions made.

Everything delivered will be evaluated: from the correctness of the analyses, assumptions, data processing, models used, clarity of the code, and the generated predictions.

## Deliverable

You must export your repository as a .zip file and send it to the interviewer one day before the interview.

Good luck.