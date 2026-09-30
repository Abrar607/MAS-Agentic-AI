import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.

dotenv.load_dotenv()

from smolagents import tool, CodeAgent, ToolCallingAgent, OpenAIServerModel

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", ""),)
)

"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""

# Helper tools:

def _resolve_item_name(query: str) -> str:
    query_lower = query.lower().strip()
    all_items = [p["item_name"] for p in paper_supplies]
    for name in all_items:
        if name.lower() == query_lower:
            return name
    candidates = []
    for name in all_items:
        name_lower=name.lower()
        if query_lower in name_lower or name_lower in query_lower:
            candidates.append(name)
    if len(candidates) == 1:
        return candidates[0]

    query_words = set(query_lower.replace("-", " ").replace(",", " ").split())

    filler = {"sheets","of","high", "quality", "in", "various","colors", "assorted",
            "size", "white", "the", "a", "an", "for", "and", "per", "rolls", "roll",
            "x", "inch", "inches", "reams", "ream", "colorful", "sturdy"}
    query_keywords = query_words - filler

    best_score = 0
    best_match = None
    for name in all_items:
        name_words = set(name.lower().replace("-", " ").replace(",", " ").split())
        overlap = len(query_keywords & name_words)
        if overlap > best_score:
            best_score = overlap
            best_match = name
    if best_score > 0:
        return best_match
    return None

# Tools for inventory agent
@tool
def check_inventory(as_of_date: str) -> str:
    """ 
    Check the full inventory status as of a given date, 
    return all items with their current stock levels and flag items that are below their minimum stock level and need reordering.
    
    Args:
        as_of_date: ISO-formatted date string (YYYY-MM-DD)to check inventory as of.
    Returns:
        formatted string listing all inventory items, their stock, min levels and reorder flags."""
    
    inventory = get_all_inventory(as_of_date)
    inventory_df = pd.read_sql("SELECT * from inventory", db_engine)

    result_lines = ["Current Inventory Status:"]
    reorder_needed = []

    for _, item in inventory_df.iterrows():
        name = item["item_name"]
        stock = inventory.get(name, 0)
        min_level = item["min_stock_level"]
        unit_price = item["unit_price"]
        status = "ok"
        if stock <= min_level:
            status = "Low - Reorder needed"
            reorder_needed.append(name)
        result_lines.append(
            f" -{name}: stock={int(stock)}, min_level={int(min_level)}, "
            f"unit_price=${unit_price:.2f}, status={status}"
        )
    if reorder_needed:
        result_lines.append(f"\nItems needing reorder: {', '.join(reorder_needed)}")
    else:
        result_lines.append("\nAll items are adequately stocked.")

    return "\n".join(result_lines)

@tool 
def check_item_stock(item_name: str, as_of_date: str) -> str:
    """ 
    Check the stock level and details for a specific item as of a given date.
    The item_name will be fuzzy-matched to the closest item in the catalog.
    Use exact catalog names when possible. Available catalog items include:
    A4 paper, Letter-sized paper, Cardstock, Colored paper, Glossy paper, Matte paper,
    Recycled paper, Eco-friendly paper, Poster paper, Banner paper, Kraft paper,
    Construction paper, Wrapping paper, Glitter paper, Decorative paper, Letterhead paper,
    Legal-sized paper, Crepe paper, Photo paper, Uncoated paper, Butcher paper, Heavyweight paper,
    Standard copy paper, Bright-colored paper, Patterned paper, Paper plates, Paper cups,
    Paper napkins, Disposable cups, Table covers, Envelopes, Sticky notes, Notepads, Invitation cards,
    Flyers, Party streamers, Decorative adhesive tape (washi tape), Paper party bags, Name tags with lanyards,
    Presentation folders, Large poster paper (24x36 inches), 
    Rolls of banner paper (36-inch width), 100 lb cover stock, 80 lb text paper,
    250 gsm cardstock, 220 gsm poster paper.

    Args:
        item_name: The name of the inventory item to check (will be fuzzy-matched).
        as_of_date: ISO-formatted date string (YYYY-MM-DD)
    Returns:
        string with item stock level, unit price, min stock level and whether reorder is needed."""
    resolved = _resolve_item_name(item_name)
    if resolved:
        item_name = resolved
    
    stock_df = get_stock_level(item_name, as_of_date)
    stock = int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0

    inv_df = pd.read_sql(
        "SELECT * FROM inventory WHERE item_name = :name",
        db_engine, params={"name": item_name}
    )

    if inv_df.empty:
        known = [p for p in paper_supplies if p["item_name"].lower() == item_name.lower()]
        if known:
            return(
                f"'{item_name}' is a known paper product "
            )
        return f"Item '{item_name}' not found in inventory or known product catalog."
    
    row = inv_df.iloc[0]
    min_level = int(row["min_stock_level"])
    unit_price = row["unit_price"]
    needs_reorder = stock <= min_level

    return (
        f"Item: {item_name}\n"
        f" Current Stock: {stock}\n"
        f" Unit Price: ${unit_price:.2f}\n"
        f" Min Stock Level: {min_level}\n"
        f" Needs Reorder: {'Yes' if needs_reorder else 'No'}"
    )

@tool
def reorder_stock(item_name: str, quantity: int, order_date: str) -> str:
    """ 
    Place a reorder for an item from the supplier. The creates a stock_orders transaction
    and deducts the cost from cash balance. Checks cash availability before ordering.
    The item_ame will be fuzzy-matched tothe closest item in the catalog.
    
    Args:
        item_name: the name of the item to reorder (fuzzy-matched to the catalog)
        quantity: number of units to order.
        order_date: ISO-formatted date string (YYYY-MM-DD) for the order.
    
    Returns:
        string confirming the reorder or explaining why it can't be fulfilled."""
    resolved = _resolve_item_name(item_name)
    if resolved:
        item_name = resolved
    
    inv_df = pd.read_sql(
        "SELECT * FROM inventory WHERE item_name= :name",
        db_engine, params={"name": item_name}
    )

    if inv_df.empty:
        known = [p for p in paper_supplies if p["item_name"].lower() == item_name.lower()]
        if not known:
            return(
                f"Error: Item '{item_name}' not found in inventory or known product catalog. Can't reorder. "
            )
        unit_price = known[0]["unit_price"]
    else:
        unit_price = inv_df.iloc[0]["unit_price"]
    
    total_cost = quantity * unit_price
    cash = get_cash_balance(order_date)

    if total_cost > cash:
        return (
            f"Insufficient funds to reorder {quantity} units of '{item_name}'"
            f"Cost: ${total_cost:.2f}, Available cash: ${cash:.2f}"
        )
    delivery_date = get_supplier_delivery_date(order_date, quantity)
    tx_id = create_transaction(item_name, "stock_orders", quantity, total_cost, delivery_date)

    return (
            f"Reorder placed!\n"
            f" Item: {item_name}\n"
            f" Quantity: {quantity}\n"
            f" Total Cost: ${total_cost:.2f}\n"
            f" Delivery Date: {delivery_date}\n"
            f" Transaction ID: {tx_id}"
        )

# Tools for quoting agent
@tool
def get_quote_history(search_terms: str) -> str:
    """
    Search historical quotes by keywords to find relevant past pricing. Helps determine competitive and consistent
    pricing for new quote requests.
    
    Args:
        search_terms: comma-seperated search keywords.
    Returns:
        formatted string of matching historical quotes with amounts and explanations."""
    terms = [t.strip() for t in search_terms.split(",") if strip()]
    results = search_quote_history(terms, limit=5)

    if not results:
        return "No matching historical quotes found."
    
    lines = [f"Found {len(results)} historical quote(s):"]
    for i, q in enumerate(results, 1):
        lines.append(
            f"\n Quote {i}:\n"
            f" Request: {q['original_request'][:150]}...\n"
            f" Total Amount: ${q['total_amount']:.2f}\n"
            f" Job Type: {q.get('job_type', 'N/A')}\n"
            f" Order Size: {q.get('order_size', 'N/A')}\n"
            f" Event Type: {q.get('event_type', 'N/A')}\n"
            f" Explanation: {q['quote_explanation'][:200]}..."
        )
    return "\n".join(lines)

@tool
def generate_quote(item_name: str, quantity: int, as_of_date: str) -> str:
    """
    Generate a prcie quote for  customer based on item, quantity nd current inventory.
    Applies bulk discounts: 5% for 100-499 units, 10% for 500-999, and 15% for 1000+.
    The item_name will be fuzzy-matched to the closest item in the catalog.
    
    Args:
        item_name: the name of the item the customer wants (fuzzy-matched to the catalog)
        quantity: the number of units requested.
        as_of_date: ISO-formatted date (YYYY-MM-DD) for the quote.
    Returns:
        formatted quote string including pricing, discounts, availability and delivery estimate."""
    resolved = _resolve_item_name(item_name)
    if resolved:
        item_name = resolved
    
    inv_df = pd.read_sql(
        "SELECT * FROM inventory WHERE item_name= :name",
        db_engine, params={"name": item_name}
    )

    if inv_df.empty:
        known = [p for p in paper_supplies if p["item_name"].lower() == item_name.lower()]
        if not known:
            return(
                f"Error: Item '{item_name}' not found in inventory or known product catalog. "
            )
        unit_price = known[0]["unit_price"]
        current_stock = 0
    else:
        unit_price = inv_df.iloc[0]["unit_price"]
        stock_df = get_stock_level(item_name, as_of_date)
        current_stock = int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0
    
    if quantity >= 1000:
        discount_pct = 15
    elif quantity >= 500:
        discount_pct = 10
    elif quantity >= 100:
        discount_pct = 5
    else:
        discount_pct = 0
    
    base_price = quantity * unit_price
    discount_amount = base_price * (discount_pct/ 100)
    final_price = base_price - discount_amount

    if current_stock >= quantity:
        availability = "In stock - can be fulfilled immediately."
        delivery = as_of_date
    elif current_stock > 0:
        shortfall = quantity - current_stock
        delivery = get_supplier_delivery_date(as_of_date, shortfall)
        availability = (
            f"Partial stock available ({current_stock} units in stock)."
            f"Remaining {shortfall} units need to be ordered."
            f"Estimated full delivery by: {delivery}"
        )
    else:
        delivery = get_supplier_delivery_date(as_of_date, quantity)
        availability = f"Out of stock. Needs to be ordered. Estimated delivery: {delivery}"
    return (
        f"--- Quote ---\n"
        f" Item: {item_name}\n"
        f" Quantity: {quantity}\n"
        f" Unit Price: ${unit_price:.2f}\n"
        f" Base Price: ${base_price:.2f}\n"
        f" Bulk Discount: {discount_pct}% (-${discount_amount:.2f})\n"
        f" Total Quote: ${final_price:.2f}\n"
        f" Availability: {availability}\n"
    )
# Tools for ordering agent
@tool
def fulfill_order(item_name: str, quantity: int, order_date: str) -> str:
    """
    Fulfill a customer order by recording a sales transaction.
    Checks stock availability ad cash implications. If stock is insufficient,
    reorders from supplier first.
    The item_name will be fuzzy-matched to the closest tem in the catalog.
    
    Args:
        item_name: the name of the item being sold (fuzzy-matched)
        quantity: the number of units the customer wants to buy
        order_date: ISO-formatted date (YYYY-MM-DD) for the sale.
    Returns:
        string confirming the order fulfillment or explaining why it can't be completed."""
    resolved = _resolve_item_name(item_name)
    if resolved:
        item_name = resolved
    
    inv_df = pd.read_sql(
        "SELECT * FROM inventory WHERE item_name= :name",
        db_engine, params={"name": item_name}
    )

    if inv_df.empty:
        known = [p for p in paper_supplies if p["item_name"].lower() == item_name.lower()]
        if not known:
            return(
                f"Error: Item '{item_name}' not found in inventory or known product catalog. "
            )
        unit_price = known[0]["unit_price"]
        current_stock = 0
    else:
        unit_price = inv_df.iloc[0]["unit_price"]
        stock_df = get_stock_level(item_name, order_date)
        current_stock = int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0
    
    if quantity >= 1000:
        discount_pct = 15
    elif quantity >= 500:
        discount_pct = 10
    elif quantity >= 100:
        discount_pct = 5
    else:
        discount_pct = 0
    
    base_price = quantity * unit_price
    discount_amount = base_price * (discount_pct/ 100)
    sale_price = base_price - discount_amount

    if current_stock < quantity:
        shortfall = quantity - current_stock
        reorder_cost = shortfall * unit_price
        cash = get_cash_balance(order_date)

        if reorder_cost > cash:
            return (
            f"Insufficient funds to reorder {quantity} units of '{item_name}'"
            f"Only {current_stock} in stock, and insufficient funds to reorder."
            f"need: ${reorder_cost:.2f}, Available cash: ${cash:.2f}"
        )
        delivery_date = get_supplier_delivery_date(order_date, quantity)
        create_transaction(item_name, "stock_orders", shortfall, reorder_cost, delivery_date)
        fulfillment_date = delivery_date
    else:
        fulfillment_date = order_date
    
    tx_id = create_transaction(item_name, "sales", quantity, sale_price, fulfillment_date)

    return (
            f"Order fulfilled\n"
            f" Item: {item_name}\n"
            f" Quantity: {quantity}\n"
            f" Unit Price: ${unit_price:.2f}\n"
            f" Discount: {discount_pct}%\n"
            f" Total Sale: ${sale_price:.2f}\n"
            f" Fulfillment Date: {fulfillment_date}\n"
            f" Transaction ID: {tx_id}"
        )

@tool
def check_cash_balance(as_of_date: str) -> str:
    """
    Check the current cash balance of the company as of the given date.
    
    Args:
        as_of_date: ISO-formatted date string (YYYY-MM-DD)
    Returns:
        string showing teh current cash balance."""
    balance = get_cash_balance(as_of_date)
    return f"Cash balance as of {as_of_date}: ${balance:.2f}"

@tool
def get_financial_summary(as_of_date: str) -> str:
    """
    Generate a finanacial summary report including cash balance, inventory value, total assets and top-selling products.
    
    Args:
        as_of_date: ISO-formatted date string (YYYY-MM-DD)
    Returns:
        formatted financial summary string."""
    report = generate_financial_report(as_of_date)
    lines = [
        f"--- Financial Report as of {as_of_date} ---",
        f" Cash Balance: ${report['cash_balance']:.2f}",
        f" Inventory Value: ${report['inventory_value']:.2f}",
        f" Total Assets: ${report['total_assets']:.2f}",
        f"\n Top Selling Products:"
    ]
    for p in report["top_selling_product"]:
        if p.get("item_name"):
            lines.append(f"   - {p['item_name']}: {p.get('total_units', 0)} units, ${p.get('total_revenue', 0):.2f} revenue")
    return "\n".join(lines)

_catalog_names = [p["item_name"] for p in paper_supplies]
_catalog_list_str = ", ".join(_catalog_names)

_agent_catalog_note = (
    "Important rules:\n"
    "1. The request includes a 'Date of request' in YYY-MM-DD format. You must use that date."
    "for all tool calls (as_of_date / order__date). Never use any other date."
    "2. Customer descriptions must be mapped to exact catalog item names. the full catalog is in:\n"
    f"{_catalog_list_str}\n"
    "3. common mappings: 'colored paper' -> 'Colored paper', 'cardstock' -> 'Cardstock',"
    "'washi tape' -> 'Decorative adhesive tape (washi tape)', 'construction paper' -> 'Construction paper',"
    "'glossy paper' -> 'Glossy paper', 'poster paper' -> 'Poster paper', 'recycled paper' -> 'Recycled paper',"
    "'copy paper' or 'printer paper' -> 'Standard copy paper',"
    "'matter paper' -> 'Matte paper', 'photo paper' -> 'Photo paper', 'napkins' -> 'Paper napkins',"
    "'cups' -> 'Paper cups', 'plates' -> 'Paper plates','poster board' or 'large poster' -> 'Large poster paper (24x36 inches)',"
    "'banner papaer rolls' -> 'Rolls of banner paper (36-inch width)', 'streamers' -> 'Party streamers', 'envelopes' -> 'Envelopes',"
    "'flyers' -> 'Flyers', 'heavyweight' or 'heavy cardstock' -> 'Heavyweight paper', 'invitation cards' -> 'Invitation cards',"
    "'notepads' -> 'Notepads'\n"
    "4. Items like 'balloons', 'tickets', 'A3 paper', 'signage cardboard' do not exist in teh catalog."
    "If an item cannot be mapped, say it's unavailable.\n"
    "5. When processing orders, generate a quote first then fulfill orders for available items.\n"
    "6. Always try to fulfill as much of the order as possible with available catalog items.\n"
)
# Set up your agents and create an orchestration agent that will manage them.
inventory_agent = ToolCallingAgent(
    tools=[check_inventory, check_item_stock, reorder_stock],
    model=model,
    name="inventory_agent",
    description=(
        "Manages inventory for the Beaver's Choice Paper company."
        "Can check current stock levels for all items or specific items."
        "identify items that need reordering, and place restock orders with the supplier."
        "Use this agent for any inventory-related questions or restocking needs."
        "Important: always pass the request date (YYYY-MM-DD) from the customer request in your task message."
    ), max_steps=6,
)

quoting_agent = ToolCallingAgent(
    tools=[get_quote_history, generate_quote, check_item_stock],
    model=model,
    name="quoting_agent",
    description=(
        "Generates price quotes for customers of the Beaver's Choice Paper Company."
        "Can search historical quotes for similar orders and generate new quotes with bulk discounts,"
        "and check item availability. Use this agent whena customer requests a quote or pricing information."
        "Important: Always pass the request date (YYYY-MM-DD) and use exact catalog item names in your task message."
    ), max_steps=6,
)

sales_agent = ToolCallingAgent(
    tools=[fulfill_order, check_item_stock, check_cash_balance, get_financial_summary],
    model=model,
    name="sales_agent",
    description=(
        "Manages sales transactions and order fulfillment for the Beaver's Choice Paper Company."
        "Can process customer orders, check stock before fulfilling, verify cash balance"
        "and generate financial summaries. Use this agent to finalize sales and complete transactions."
        "Important: Always pass the request date (YYYY-MM-DD) and use exact catalog items names in your task message."
    ), max_steps=8,
)

orchestrator_agent = ToolCallingAgent(
    tools=[],
    model=model,
    managed_agents=[inventory_agent, quoting_agent, sales_agent],
    name="orchestrator_agent",
    description=(
        "Main orchestrator that delegates customer requests to the corresponding agents."
        + _agent_catalog_note +
        "\nWorkflow for each customer request:\n"
        "1. First, ask the quoting_agent to generate quotes for each requested item."
        "Always pass the exact request date (from 'Date of request': YYYY-MM-DD) to the agent."
        "2. Then, ask the sales_agent to fulfill orders for items that can be fulfilled."
        "Pass the full request text including the date.\n"
        "3. If items need restocking first, ask inventory_agent to reorder before fulfilling."
        "4. Compile a customer-facing response with: qupted prices, order status, delivery statues,"
        "also which items couldn't be fulfilled and why\n"
        "Important: Always extract and pass the 'Date of request: YYYY-MM-DD' to every agent task.\n"
        "Important: Map customer item description to catalog names before delegating.\n"
    ), max_steps=10,
)
# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        try:
            response = orchestrator_agent.run(request_with_date)
            status = "error" if "can't" in str(response).lower() and "not available" in str(response).lower() else "fulfilled"
        except Exception as e:
            print(f"Error processing request: {e}")
            response = f"Error: {str(e)}"
            status = "error"

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
