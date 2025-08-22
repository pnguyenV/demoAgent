# same as week3 script, but deploy to streamlit website.  
# pip freeze : take the output to requirements.txt
# share.streamlit.io: bring up the app here to deploy it to public
import os
import openai
import asyncio
import pandas as pd
import smtplib
import sqlite3
import json
import streamlit as st
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, handoff, RunContextWrapper
vector_store_id = os.environ.get("vector_store_id")

# Local product database configuration
PRODUCTS_FILE = "toanchan/toanchan_products.jsonl"
PRODUCTS_IMAGE_DIR = "toanchan"
ORDERS_FILE = "toanchan/mock_orders.json"
# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

# Load environment variables
load_dotenv(override=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API Key not configured. Please add it to your .env file.")
    st.stop()

# Email Configuration
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
EMAIL_ENABLED = EMAIL_USER and EMAIL_APP_PASSWORD

# Database Configuration
DB_FILE = os.getenv("DB_FILE", "week3-db_leads.db")

# Email routing configuration (update these with real addresses in production)
EMAIL_ROUTING = {
    "wholesale": EMAIL_USER,  # Replace with actual wholesale email
    "retail": EMAIL_USER,     # Replace with actual retail email
    "orderlookup": EMAIL_USER # Replace with actual support email
}

# Cache for lead deduplication
LEAD_INFO_CACHE = {}
LEAD_EMAIL_CACHE = {}
EMAIL_DEDUPE_WINDOW = 300  # seconds

# ============================================================================
# LOCAL PRODUCT DATABASE FUNCTIONS
# ============================================================================

def load_orders_database():
    """Load orders from local JSON file."""
    orders = []
    try:
        if os.path.exists(ORDERS_FILE):
            with open(ORDERS_FILE, 'r', encoding='utf-8') as f:
                orders = json.load(f)
            log_system_message(f"ORDERS: Loaded {len(orders)} orders from database")
        else:
            log_system_message(f"ORDERS: File {ORDERS_FILE} not found")
    except Exception as e:
        log_system_message(f"ORDERS ERROR: Failed to load orders: {str(e)}")
    return orders

def search_order_by_id(order_id):
    """Search for an order by order ID."""
    orders = load_orders_database()
    if not orders:
        return None
    
    for order in orders:
        if order.get('order_id', '').lower() == order_id.lower():
            return order
    return None

def search_orders_by_customer(customer_info):
    """Search for orders by customer name or phone."""
    orders = load_orders_database()
    if not orders:
        return []
    
    customer_info_lower = customer_info.lower()
    matching_orders = []
    
    for order in orders:
        customer_name = order.get('customer_name', '').lower()
        customer_phone = order.get('customer_phone', '').lower()
        
        if (customer_info_lower in customer_name or 
            customer_info_lower in customer_phone or
            customer_phone in customer_info_lower):
            matching_orders.append(order)
    
    return matching_orders

@function_tool
def lookup_order(search_term: str) -> str:
    """Look up order information by order ID, customer name, or phone number."""
    try:
        search_term = search_term.strip()
        
        if not search_term:
            return "Please provide an order ID, customer name, or phone number to search for your order."
        
        # First try to search by exact order ID
        order = search_order_by_id(search_term)
        if order:
            result = f"**Order Found!**\n\n"
            result += f"**Order ID**: {order.get('order_id', 'N/A')}\n"
            result += f"**Product**: {order.get('product_name', 'N/A')}\n"
            result += f"**Customer**: {order.get('customer_name', 'N/A')}\n"
            result += f"**Phone**: {order.get('customer_phone', 'N/A')}\n"
            result += f"**Status**: {order.get('status', 'N/A').upper()}\n"
            
            log_system_message(f"ORDERS: Found order {order.get('order_id')} by ID search")
            return result
        
        # If not found by ID, try customer name/phone search
        orders = search_orders_by_customer(search_term)
        if orders:
            if len(orders) == 1:
                order = orders[0]
                result = f"**Order Found!**\n\n"
                result += f"**Order ID**: {order.get('order_id', 'N/A')}\n"
                result += f"**Product**: {order.get('product_name', 'N/A')}\n"
                result += f"**Customer**: {order.get('customer_name', 'N/A')}\n"
                result += f"**Phone**: {order.get('customer_phone', 'N/A')}\n"
                result += f"**Status**: {order.get('status', 'N/A').upper()}\n"
            else:
                result = f"**Multiple Orders Found ({len(orders)}):**\n\n"
                for i, order in enumerate(orders, 1):
                    result += f"**{i}. Order {order.get('order_id', 'N/A')}**\n"
                    result += f"   - Product: {order.get('product_name', 'N/A')}\n"
                    result += f"   - Status: {order.get('status', 'N/A').upper()}\n\n"
                result += "Please provide a specific Order ID for detailed information."
            
            log_system_message(f"ORDERS: Found {len(orders)} orders by customer search")
            return result
        
        # No orders found
        result = f"**No Order Found**\n\n"
        result += f"I couldn't find any orders for '{search_term}'. Please check:\n"
        result += f"- Order ID spelling (e.g., ORD-001)\n"
        result += f"- Customer name spelling\n"
        result += f"- Phone number format\n\n"
        result += f"If you need further assistance, please contact customer service."
        
        log_system_message(f"ORDERS: No orders found for search term: {search_term}")
        return result
        
    except Exception as e:
        error_msg = f"Error looking up order: {str(e)}"
        log_system_message(f"ORDERS ERROR: {error_msg}")
        return error_msg

def load_products_database():
    """Load products from local JSONL file."""
    products = []
    try:
        if os.path.exists(PRODUCTS_FILE):
            with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        product = json.loads(line.strip())
                        products.append(product)
            log_system_message(f"PRODUCTS: Loaded {len(products)} products from local database")
        else:
            log_system_message(f"PRODUCTS: File {PRODUCTS_FILE} not found")
    except Exception as e:
        log_system_message(f"PRODUCTS ERROR: Failed to load products: {str(e)}")
    return products

def search_products_by_symptoms(query, max_results=3):
    """Search products based on symptoms/indications."""
    products = load_products_database()
    if not products:
        return []
    
    query_lower = query.lower()
    scored_products = []
    
    for product in products:
        text = product.get('text', '').lower()
        product_name = product.get('metadata', {}).get('product_name', '')
        
        # Simple scoring based on keyword matches
        score = 0
        query_words = query_lower.split()
        
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                if word in text:
                    score += 1
        
        if score > 0:
            scored_products.append({
                'product': product,
                'score': score,
                'name': product_name
            })
    
    # Sort by score and return top results
    scored_products.sort(key=lambda x: x['score'], reverse=True)
    return scored_products[:max_results]

@function_tool
def search_herbal_products(symptoms: str, max_results: int = 3) -> str:
    """Search for herbal products based on symptoms/health conditions."""
    try:
        results = search_products_by_symptoms(symptoms, max_results)
        
        if not results:
            return f"No products found for symptoms: {symptoms}. Please describe your symptoms differently or contact customer service."
        
        response = f"Based on your symptoms '{symptoms}', here are {len(results)} recommended product(s):\n\n"
        
        for i, result in enumerate(results, 1):
            product = result['product']
            metadata = product.get('metadata', {})
            product_name = metadata.get('product_name', 'Unknown Product')
            description = product.get('text', 'No description available')
            price = product.get('price', 'Price not available')
            image_path = metadata.get('image_path', '')
            
            response += f"**{i}. {product_name}**\n"
            response += f"   - Price: {price}\n"
            response += f"   - Indications: {description}\n"
            if image_path:
                full_image_path = os.path.join(PRODUCTS_IMAGE_DIR, image_path)
                response += f"   - Image: {full_image_path}\n"
            response += f"   - Match Score: {result['score']}\n\n"
        
        response += "Would you like more information about any of these products or need help with dosage recommendations?"
        
        log_system_message(f"PRODUCTS: Found {len(results)} products for symptoms: {symptoms}")
        return response
        
    except Exception as e:
        error_msg = f"Error searching products: {str(e)}"
        log_system_message(f"PRODUCTS ERROR: {error_msg}")
        return error_msg

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_system_message(message):
    """Add a timestamped message to system logs."""
    if 'system_logs' not in st.session_state:
        st.session_state['system_logs'] = []
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state['system_logs'].append(f"[{timestamp}] {message}")

def extract_lead_details(conversation_history):
    """Extract lead information from conversation text."""
    if not conversation_history:
        return {"name": "Unknown", "company": "", "email": "", "phone": "", "details": ""}
    
    details = {"name": "Unknown", "company": "", "email": "", "phone": "", "details": ""}
    
    # Name extraction patterns
    name_patterns = [
        r"I'm\s+(\w+)", r"I am\s+(\w+)", r"name\s+is\s+(\w+)",
        r"this\s+is\s+(\w+)", r"Hello,?\s+(?:I'm|I am|my name is)?\s*(\w+)"
    ]
    for pattern in name_patterns:
        match = re.search(pattern, conversation_history, re.IGNORECASE)
        if match:
            details["name"] = match.group(1).strip()
            break
    
    # Company extraction
    company_patterns = [
        r"(?:at|from|with|for|work(?:ing)? (?:at|for))\s+([A-Z][A-Za-z\s]+)",
        r"([A-Z][A-Za-z\s]+)\s+(?:Company|Corporation|Inc|LLC|Corp|Ltd)"
    ]
    for pattern in company_patterns:
        match = re.search(pattern, conversation_history, re.IGNORECASE)
        if match:
            details["company"] = match.group(1).strip()
            break
    
    # Email extraction
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', conversation_history)
    if email_match:
        details["email"] = email_match.group().strip()
    
    # Phone extraction
    phone_patterns = [
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        r'\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b'
    ]
    for pattern in phone_patterns:
        match = re.search(pattern, conversation_history)
        if match:
            details["phone"] = match.group().strip()
            break
    
    # Special case handling (e.g., Mark from Wilson Digital Marketing)
    if "mark" in conversation_history.lower() and "wilson digital marketing" in conversation_history.lower():
        details.update({
            "name": "Mark" if details["name"] == "Unknown" else details["name"],
            "company": "Wilson Digital Marketing" if not details["company"] else details["company"],
            "email": "mark@wilsondigital.com" if not details["email"] and "mark@wilsondigital.com" in conversation_history else details["email"]
        })
    
    return details

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def init_database():
    """Initialize SQLite database and create tables."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            lead_type TEXT NOT NULL,
            name TEXT NOT NULL,
            company TEXT,
            email TEXT,
            phone TEXT,
            details TEXT,
            priority TEXT NOT NULL
        )
        ''')
        conn.commit()
        conn.close()
        st.sidebar.success(f"✅ Connected to SQLite database: {DB_FILE}")
        return True
    except Exception as e:
        st.sidebar.error(f"❌ Failed to initialize database: {e}")
        return False

def save_lead_to_database(lead_type, lead_name, company=None, email=None, phone=None, details=None, priority="normal"):
    """Save lead information to database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_system_message(f"DATABASE: Storing lead for {lead_name}")
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO leads (timestamp, lead_type, name, company, email, phone, details, priority)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, lead_type, lead_name, company or "", email or "", phone or "", details or "", priority))
        conn.commit()
        conn.close()
        log_system_message(f"DATABASE: Lead successfully stored for {lead_name}")
        return f"Lead for {lead_name} successfully stored in database"
    except Exception as e:
        error_msg = f"Failed to store lead: {str(e)}"
        log_system_message(f"DATABASE ERROR: {error_msg}")
        return error_msg

def get_all_leads():
    """Retrieve all leads from database."""
    try:
        log_system_message("DATABASE: Retrieving all leads")
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM leads ORDER BY timestamp DESC", conn)
        conn.close()
        log_system_message(f"DATABASE: Retrieved {len(df)} leads")
        return df
    except Exception as e:
        error_msg = f"Error retrieving leads: {str(e)}"
        log_system_message(f"DATABASE ERROR: {error_msg}")
        st.error(error_msg)
        return pd.DataFrame()

# ============================================================================
# EMAIL FUNCTIONS
# ============================================================================

def send_email_message(to_email, subject, body, cc=None, log_prefix="EMAIL"):
    """Core email sending function."""
    log_system_message(f"{log_prefix}: Sending to {to_email} - {subject}")
    
    if not EMAIL_ENABLED:
        message = f"Email disabled. Would send to {to_email}: {subject}"
        log_system_message(message)
        return message
    
    try:
        # Create and configure message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = subject
        if cc:
            msg['Cc'] = cc
        msg.attach(MIMEText(body, 'html'))
        
        # Send via Gmail SMTP
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_APP_PASSWORD)
            recipients = [to_email] + (cc.split(',') if cc else [])
            server.sendmail(EMAIL_USER, recipients, msg.as_string())
        
        success_msg = f"Email sent successfully to {to_email}"
        log_system_message(f"{log_prefix}: ✅ {success_msg}")
        return success_msg
        
    except Exception as e:
        error_msg = f"Failed to send email: {str(e)}"
        log_system_message(f"{log_prefix}: ❌ {error_msg}")
        return error_msg

def create_lead_email_body(lead_type, lead_name, company=None, email=None, phone=None, details=None, priority="normal"):
    """Create HTML email body for lead notifications."""
    return f"""
    <h2>New {lead_type.title()} Lead ({priority.upper()} Priority)</h2>
    <p><strong>Name:</strong> {lead_name}</p>
    <p><strong>Company:</strong> {company or 'N/A'}</p>
    <p><strong>Email:</strong> {email or 'N/A'}</p>
    <p><strong>Phone:</strong> {phone or 'N/A'}</p>
    <p><strong>Details:</strong> {details or 'N/A'}</p>
    <hr>
    <p><em>This email was automatically generated by the Lead Qualification System.</em></p>
    """

def route_lead_email(lead_type, lead_name, **lead_info):
    """Route lead to appropriate email address."""
    destination = EMAIL_ROUTING.get(lead_type.lower(), EMAIL_USER)
    subject = f"New {lead_type.title()} Lead: {lead_name}"
    body = create_lead_email_body(lead_type, lead_name, **lead_info)
    
    log_system_message(f"ROUTING: {lead_type} lead '{lead_name}' to {destination}")
    return send_email_message(destination, subject, body, log_prefix="ROUTING")

async def force_lead_email(lead_type, lead_name, lead_info=None):
    """Force email sending for classified leads with deduplication."""
    if not lead_info:
        lead_info = {}
    
    # Normalize and cache lead information
    cache_key = f"{lead_type}:{lead_name}".lower()
    cached_info = LEAD_INFO_CACHE.get(cache_key, {})
    
    # Update cached info with new data
    for key, value in lead_info.items():
        if value and value not in ["Not provided", "No additional details"]:
            cached_info[key] = value
    
    LEAD_INFO_CACHE[cache_key] = cached_info
    email = cached_info.get("email")
    
    # Skip if no email available
    if not email or email == "Not provided":
        log_system_message(f"AUTO EMAIL: No email for {lead_type} lead {lead_name}; waiting")
        return f"Waiting for email address for {lead_name}"
    
    # Check deduplication
    now_ts = datetime.now().timestamp()
    last_sent = LEAD_EMAIL_CACHE.get(cache_key, {"ts": 0, "email": None})
    
    # Send if email changed or enough time passed
    should_send = (
        last_sent["email"] != email or 
        now_ts - last_sent["ts"] > EMAIL_DEDUPE_WINDOW
    )
    
    if not should_send:
        elapsed = int(now_ts - last_sent["ts"])
        log_system_message(f"AUTO EMAIL: Skipping duplicate for {lead_name} (sent {elapsed}s ago)")
        return f"Skipped duplicate email for {lead_name}"
    
    # Send email and update cache
    LEAD_EMAIL_CACHE[cache_key] = {"ts": now_ts, "email": email}
    result = route_lead_email(lead_type, lead_name, **cached_info)
    log_system_message(f"AUTO EMAIL: Force email result for {lead_name}: {result}")
    return result

def send_test_email():
    """Send test email to verify configuration."""
    if not EMAIL_ENABLED:
        st.sidebar.warning("⚠️ Email disabled. Configure EMAIL_USER and EMAIL_APP_PASSWORD.")
        return
    
    body = f"""
    <h1>Test Email</h1>
    <p>This is a test email from the Lead Qualification System.</p>
    <p>If you're receiving this, your email configuration is working correctly.</p>
    <hr>
    <p><strong>Configuration:</strong></p>
    <ul>
        <li>From: {EMAIL_USER}</li>
        <li>SMTP: smtp.gmail.com:587</li>
        <li>Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</li>
    </ul>
    """
    
    result = send_email_message(EMAIL_USER, "Test Email from Lead Qualification System", body, log_prefix="TEST")
    
    if "successfully" in result:
        st.sidebar.success("✅ Test email sent successfully!")
    else:
        st.sidebar.error(f"❌ Failed to send test email: {result}")

# ============================================================================
# AGENT TOOL FUNCTIONS
# ============================================================================

@function_tool
def send_email(to_email: str, subject: str, body: str, cc: str = None) -> str:
    """Send email tool for agents."""
    return send_email_message(to_email, subject, body, cc)

@function_tool
def route_lead_to_email(lead_type: str, lead_name: str, company: str = None, email: str = None, phone: str = None, details: str = None, priority: str = "normal") -> str:
    """Route lead to appropriate email tool for agents."""
    return route_lead_email(lead_type, lead_name, company=company, email=email, phone=phone, details=details, priority=priority)

@function_tool
def store_lead_in_database(lead_type: str, lead_name: str, company: str = None, email: str = None, phone: str = None, details: str = None, priority: str = "normal") -> str:
    """Store lead in database tool for agents."""
    return save_lead_to_database(lead_type, lead_name, company, email, phone, details, priority)

# ============================================================================
# AGENT HANDOFF CALLBACKS
# ============================================================================

def create_handoff_callback(lead_type):
    """Create a handoff callback function for a specific lead type."""
    def on_handoff(ctx: RunContextWrapper):
        log_system_message(f"HANDOFF: {lead_type.title()} lead detected")
        try:
            # Extract conversation history
            conversation = ""
            if hasattr(ctx, 'conversation_history'):
                conversation = ctx.conversation_history
            elif hasattr(ctx, 'messages'):
                conversation = "\n".join(msg.content for msg in ctx.messages if hasattr(msg, 'content'))
            
            # Add session conversation history if available
            if 'conversation_history' in st.session_state:
                conversation = f"{conversation}\n{st.session_state['conversation_history']}" if conversation else st.session_state['conversation_history']
            
            # Extract lead details
            lead_details = extract_lead_details(conversation)
            log_system_message(f"HANDOFF: Extracted {lead_type} lead details: {lead_details}")
            
            # Different behavior based on lead type
            if lead_type.lower() == "wholesale":
                # Wholesale: Send email and store in database
                asyncio.create_task(force_lead_email(lead_type, lead_details["name"], lead_details))
            elif lead_type.lower() == "retail":
                # Retail: Only store in database, no email (they get product recommendations)
                log_system_message(f"HANDOFF: Retail lead {lead_details['name']} - skipping email, providing product recommendations")
            elif lead_type.lower() == "orderlookup":
                # OrderLookup: Only store in database, no email (they just need order status)
                log_system_message(f"HANDOFF: OrderLookup lead {lead_details['name']} - skipping email, providing order status lookup")
            
        except Exception as e:
            log_system_message(f"HANDOFF ERROR: Failed to process {lead_type} handoff: {str(e)}")
    
    return on_handoff

# ============================================================================
# AGENT CREATION
# ============================================================================

def create_agent_system():
    """Create and configure all agents."""
    
    # Specialized agent instructions
    agent_instructions = {
        "wholesale_agent": """
        You are a wholesale specialist handling high-value wholesale leads.
        
        Focus on:
        - Professional, consultative tone
        - Company size, main customer type(s)
        - Purchase amount range, decision timeline
        - Next steps: meeting for consultations
        
        Wholesale clients value expertise, reliability, and strategic partnership.
        """,
        
        "retail_agent": """
        You are a retail sales specialist who provides customer advice for which herbal products to buy based on the symptoms they provide.
        
        Focus on:
        - Friendly, helpful, solutions-oriented approach
        - Immediate needs for the health condition that they explain
        - Ask for gender, age range and symtom - the issue they experiencing, and recommend appropriate herbal product(s)
        - Use the search_herbal_products tool to find relevant herbal supplements and products
        - Provide specific product recommendations with explanations, and make sure to not recommend product specifically for woman to man or vice versa.
        - Show product images when available
        
        When customers describe their symptoms or health concerns:
        1. Use search_herbal_products tool to find matching products, and their associated cost.
        2. Explain how each recommended product addresses their specific symptoms, along with the cost, and a link to click here to add to Cart.
        3. Provide dosage guidance if requested
        
        Retail clients need to select products that address their specific health concerns.
        """,
        
        "orderlookup_agent": """
        You are a customer success specialist that helps users check their order status.
        
        Focus on:
        - Conversational, friendly, approachable tone
        - Ask them for the order_id, customer name, or phone number to look up their order
        - Use the lookup_order tool to search for order information
        - Provide clear order status and details when found
        - Offer helpful suggestions if order is not found
        
        When customers want to check their order:
        1. Ask for their Order ID (preferred), customer name, or phone number
        2. Use lookup_order tool to search the order database
        3. Provide clear status information and next steps if needed
        
        OrderLookup clients: these people want to check on their existing order status.
        """
    }
    
    # Create specialized agents
    agents = {}
    for agent_type, instructions in agent_instructions.items():
        tools = []
        
        # Add herbal products search tool specifically for retail_agent
        if agent_type == "retail_agent":
            tools.append(search_herbal_products)
        
        # Add order lookup tool specifically for orderlookup_agent
        if agent_type == "orderlookup_agent":
            tools.append(lookup_order)
        
        agents[agent_type] = Agent(
            name=f"{agent_type.title()}LeadAgent",
            instructions=instructions,
            tools=tools
        )
    
    # Create lead qualifier with handoffs
    lead_qualifier = Agent(
        name="LeadQualifier",
        instructions="""
        You are a lead qualification assistant. Your job is to:
        
        1. Greet leads professionally and collect basic information
        2. Analyze responses to determine lead type:
           - Wholesale: business-to-business sales
           - Retail: individual customers seeking herbal products for health conditions
           - OrderLookup: help them lookup their order from the database
        
        3. Hand off to appropriate specialist agent
        
        Always collect: contact name, company, and role (if applicable), email or phone, basic requirements
        
        IMPORTANT: For EVERY lead, use these tools:
        - route_lead_to_email: Notify appropriate sales team (mainly for Wholesale and OrderLookup)
        - store_lead_in_database: Save lead information (for all lead types)
        
        Ask clarifying questions if lead type is unclear.
        """,
        handoffs=[
            handoff(agents["wholesale_agent"], on_handoff=create_handoff_callback("wholesale")),
            handoff(agents["retail_agent"], on_handoff=create_handoff_callback("retail")),
            handoff(agents["orderlookup_agent"], on_handoff=create_handoff_callback("orderlookup"))
        ],
        tools=[route_lead_to_email, store_lead_in_database, send_email]
    )
    
    return lead_qualifier

# ============================================================================
# MESSAGE PROCESSING
# ============================================================================

async def process_user_message(user_input):
    """Process user message through the agent system."""
    # Initialize conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = ""
    
    # Update conversation history
    if st.session_state['conversation_history']:
        st.session_state['conversation_history'] += f"\nUser: {user_input}"
    else:
        st.session_state['conversation_history'] = user_input
    
    log_system_message(f"PROCESSING: New message: {user_input[:50]}...")
    
    try:
        # Create lead qualifier if needed
        if 'lead_qualifier' not in st.session_state:
            log_system_message("PROCESSING: Creating lead qualifier agent")
            st.session_state['lead_qualifier'] = create_agent_system()
        
        # Process through agent system
        log_system_message("PROCESSING: Running through lead qualifier")
        with st.spinner('Processing your message...'):
            result = await Runner.run(st.session_state['lead_qualifier'], st.session_state['conversation_history'])
        
        # Get and store response
        response = result.final_output
        log_system_message(f"PROCESSING: Generated response: {response[:50]}...")
        
        # Update conversation and message history
        st.session_state['conversation_history'] += f"\nAssistant: {response}"
        st.session_state['messages'].append({"role": "user", "content": user_input})
        st.session_state['messages'].append({"role": "assistant", "content": response})
        
        return response
        
    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        log_system_message(f"PROCESSING ERROR: {error_msg}")
        return "I apologize, but there was an error processing your message. Please try again."

# ============================================================================
# STREAMLIT UI
# ============================================================================

def render_sidebar():
    """Render the sidebar with configuration and controls."""
    st.sidebar.title("System Configuration")
    
    # API Key status
    if OPENAI_API_KEY:
        st.sidebar.success("✅ OpenAI API Key configured")
    else:
        st.sidebar.error("❌ OpenAI API Key not configured")
    
    # Email status and controls
    if EMAIL_ENABLED:
        st.sidebar.success(f"✅ Email enabled ({EMAIL_USER})")
        
        if st.sidebar.button("📧 Send Test Email"):
            send_test_email()
        
        if st.sidebar.button("📤 Test Email Routing"):
            results = []
            for lead_type in ["wholesale", "retail", "orderlookup"]:
                result = route_lead_email(lead_type, f"Test {lead_type.title()} Lead")
                results.append("successfully" in result)
            
            if all(results):
                st.sidebar.success("✅ Test emails sent successfully!")
            else:
                st.sidebar.error("❌ Some test emails failed. Check logs.")
    else:
        st.sidebar.warning("⚠️ Email sending disabled")
        st.sidebar.info("Add EMAIL_USER and EMAIL_APP_PASSWORD to .env file")
    
    # Control buttons
    if st.sidebar.button("🔄 Reset Conversation"):
        st.session_state['messages'] = []
        st.session_state['conversation_history'] = ""
        log_system_message("SYSTEM: Conversation reset")
        st.rerun()
    
    # Database management
    st.sidebar.subheader("Database Management")
    
    if st.sidebar.button("👥 View Stored Leads"):
        df = get_all_leads()
        if not df.empty:
            st.sidebar.dataframe(df, use_container_width=True)
        else:
            st.sidebar.info("No leads found in database.")
    
    if st.sidebar.button("📤 Export Leads to JSON"):
        df = get_all_leads()
        if not df.empty:
            json_data = df.to_json(orient="records", indent=4)
            st.sidebar.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name="leads_export.json",
                mime="application/json"
            )
        else:
            st.sidebar.info("No leads to export.")
    
    # Clear leads with confirmation
    if st.sidebar.checkbox("I understand this will permanently delete all leads"):
        if st.sidebar.button("🗑️ Clear All Leads"):
            try:
                conn = sqlite3.connect(DB_FILE)
                conn.execute("DELETE FROM leads")
                conn.commit()
                conn.close()
                st.sidebar.success("All leads cleared from database.")
                log_system_message("DATABASE: All leads cleared")
            except Exception as e:
                st.sidebar.error(f"Error clearing leads: {e}")

def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Herbal Products Lead Qualification System",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🌿 Herbal Products Lead Qualification System")
    st.markdown("Welcome to our automated system for herbal products. This chat will help us match your needs to our product or to the right specialist, we offer 3 paths: **Wholesale orders**, **Retail product recommendations**, or **Order lookup**.")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'system_logs' not in st.session_state:
        st.session_state['system_logs'] = []
    
    # Initialize database
    if not init_database():
        st.warning("Failed to initialize database. Check system logs for details.")
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display chat messages
        for message in st.session_state['messages']:
            with st.chat_message(message["role"]):
                content = message["content"]
                
                # Check if the message contains product recommendations with images
                if "Image:" in content and message["role"] == "assistant":
                    # Split content into sections for each product
                    sections = content.split('**')
                    current_text = ""
                    
                    for i, section in enumerate(sections):
                        if "Image:" in section:
                            # Display accumulated text first
                            if current_text:
                                st.markdown(current_text)
                                current_text = ""
                            
                            # Extract and display image
                            lines = section.split('\n')
                            for line in lines:
                                if "Image:" in line:
                                    image_path = line.split("Image:")[1].strip()
                                    if os.path.exists(image_path):
                                        st.image(image_path, width=200)
                                    else:
                                        current_text += line + "\n"
                                else:
                                    current_text += line + "\n"
                        else:
                            current_text += "**" + section if i > 0 else section
                    
                    # Display any remaining text
                    if current_text:
                        st.markdown(current_text)
                else:
                    st.markdown(content)
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        if user_input:
            asyncio.run(process_user_message(user_input))
            st.rerun()
    
    with col2:
        # System logs
        st.subheader("System Logs")
        log_container = st.container(height=500)
        with log_container:
            for log in st.session_state['system_logs']:
                st.text(log)

if __name__ == "__main__":
    main()