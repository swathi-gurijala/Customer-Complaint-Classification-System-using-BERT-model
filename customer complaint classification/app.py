from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash
import sqlite3
import os
import torch
import evaluate
from transformers import BertTokenizer, BertForSequenceClassification
import json
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback_secret_key")  # Change in production

# Ensure uploads folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Accuracy Metric
metric = evaluate.load("accuracy")

# Load BERT Model & Tokenizer
model, tokenizer, category_mapping = None, None, {}

def load_model():
    """ Load the trained BERT model and category mapping """
    global model, tokenizer, category_mapping
    try:
        model = BertForSequenceClassification.from_pretrained("./complaint_model")
        tokenizer = BertTokenizer.from_pretrained("./complaint_model")
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model, tokenizer = None, None

    try:
        with open("category_mapping.json") as f:
            category_mapping = json.load(f)
        category_mapping = {int(v): k for k, v in category_mapping.items()}
        print("✅ Category mapping loaded successfully")
    except Exception as e:
        print(f"❌ Error loading category mapping: {e}")
        category_mapping = {}

load_model()

def classify_complaint(text):
    """Classify user complaint using the BERT model"""
    if model is None or tokenizer is None:
        return "Model not loaded"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    
    category_id = torch.argmax(outputs.logits, dim=1).item()
    return category_mapping.get(category_id, "Unknown")

@app.route("/")
def home():
    return render_template("index.html")  # Ensure it correctly renders index.html

# ✅ User Signup
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form["full_name"]
        email = request.form["email"]
        password = request.form["password"]
        address = request.form["address"]
        mobile = request.form["mobile"]

        hashed_password = generate_password_hash(password)

        try:
            with sqlite3.connect("complaints.db") as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (full_name, email, password, address, mobile) 
                    VALUES (?, ?, ?, ?, ?)""",
                    (full_name, email, hashed_password, address, mobile))
                conn.commit()
            flash("✅ Account created successfully! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("❌ Email already exists. Try another email!", "error")
        
    return render_template("signup.html")

# ✅ User Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        with sqlite3.connect("complaints.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, password FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()

        if user and check_password_hash(user[1], password):
            session["user_id"] = user[0]
            session["email"] = email
            flash("✅ Login successful!", "success")
            return redirect(url_for("complaint_form"))
        else:
            flash("❌ Invalid email or password.", "error")

    return render_template("login.html")

# ✅ User Logout
@app.route("/logout")
def logout():
    session.clear()
    flash("✅ Logged out successfully!", "success")
    return redirect(url_for("login"))

# ✅ Complaint Submission
@app.route("/complaint_form", methods=["GET", "POST"])
def complaint_form():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        complaint_text = request.form.get("complaint", "").strip()
        product_rating = int(request.form.get("product_rating", 0))
        service_rating = int(request.form.get("service_rating", 0))
        product_photo = request.files.get("product_photo")

        if not complaint_text:
            return render_template("success.html", message="❌ Complaint cannot be empty!")

        filename = ""
        if product_photo and product_photo.filename:
            filename = secure_filename(product_photo.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            product_photo.save(file_path)

        try:
            category = classify_complaint(complaint_text)

            with sqlite3.connect("complaints.db") as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO complaints (user_id, text, category, product_rating, service_rating, product_photo, submission_date) 
                    VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """, (session["user_id"], complaint_text, category, product_rating, service_rating, filename))
                conn.commit()

            return render_template("success.html", message="✅ Complaint submitted successfully!")

        except sqlite3.Error as e:
            return render_template("success.html", message=f"❌ Database Error: {e}")

    return render_template("complaint_form.html")

# ✅ Admin Dashboard (View Complaints)
@app.route("/admin_dashboard")
def admin_dashboard():
    if "admin" not in session:
        flash("❌ Unauthorized access!", "error")
        return redirect(url_for("admin_login"))

    with sqlite3.connect("complaints.db") as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Fetch complaints with user details (JOIN users table)
        cursor.execute("""
           SELECT complaints.id, complaints.text, complaints.category, 
           users.full_name, users.email, users.address, users.mobile, 
           complaints.product_rating, complaints.submission_date FROM complaints JOIN users ON complaints.user_id = users.id
        """)





        complaints = cursor.fetchall()

        # Fetch registered users
        cursor.execute("SELECT id, full_name, email FROM users")
        users = cursor.fetchall()

    return render_template("admin_dashboard.html", complaints=complaints, users=users)


# ✅ View Complaint Details
@app.route('/user_complaint_details/<int:complaint_id>')
def user_complaint_details(complaint_id):
    if "admin" not in session:
        flash("❌ Unauthorized access!", "error")
        return redirect(url_for("admin_login"))

    conn = sqlite3.connect("complaints.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT u.full_name, u.email, u.address, u.mobile, 
               c.product_rating, c.service_rating, c.product_photo, 
               c.text, c.category, c.submission_date
        FROM complaints c
        JOIN users u ON c.user_id = u.id
        WHERE c.id = ?
    """, (complaint_id,))

    data = cursor.fetchone()
    conn.close()

    if not data:
        return "Complaint Not Found", 404

    user = {
        "full_name": data[0],
        "email": data[1],
        "address": data[2],
        "mobile": data[3],
        "rating": data[4] if data[4] else "N/A",
        "product_photo": data[6] if data[6] else None
    }
    
    complaint = {
        "text": data[7],
        "category": data[8],
        "submission_date": data[9]
    }

    return render_template("user_complaint_details.html", user=user, complaint=complaint)


# ✅ Delete User (Admin Only)
@app.route("/delete_user", methods=["POST"])
def delete_user():
    if "admin" not in session:
        flash("❌ Unauthorized access!", "error")
        return redirect(url_for("admin_login"))

    user_id = request.form.get("user_id")

    if not user_id:
        flash("❌ No user selected for deletion.", "error")
        return redirect(url_for("admin_dashboard"))

    try:
        with sqlite3.connect("complaints.db") as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            cursor.execute("DELETE FROM complaints WHERE user_id = ?", (user_id,))
            conn.commit()

        flash("✅ User deleted successfully!", "success")
    except sqlite3.Error as e:
        flash(f"❌ Database error: {e}", "error")

    return redirect(url_for("admin_dashboard"))


# ✅ Hardcoded Admin Credentials
ADMIN_CREDENTIALS = {
    "username": "swathi",
    "password": "Swathi1403"
}

# ✅ Admin Login (Using Hardcoded Credentials)
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == ADMIN_CREDENTIALS["username"] and password == ADMIN_CREDENTIALS["password"]:
            session["admin"] = username
            flash("✅ Admin login successful!", "success")
            print("✅ Session Admin Set:", session.get("admin"))  # Debugging
            return redirect(url_for("admin_dashboard"))
        else:
            flash("❌ Invalid credentials.", "error")
            print("❌ Login failed")

    return render_template("admin_login.html")


# ✅ Admin Logout
@app.route("/admin_logout")
def admin_logout():
    session.pop("admin", None)  # Remove admin session
    flash("✅ Admin logged out successfully!", "success")
    return redirect(url_for("home"))  # Redirect to the homepage (index.html)


if __name__ == "__main__":
    app.run(debug=True)
