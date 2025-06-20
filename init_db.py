import sqlite3

conn = sqlite3.connect("complaints.db")
cursor = conn.cursor()

# Users table (Updated: Removed rating & product_photo)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        address TEXT NOT NULL,
        mobile TEXT NOT NULL
    )
""")

# Complaints table (Updated: Added product_rating, service_rating, product_photo)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        text TEXT NOT NULL,
        category TEXT NOT NULL,
        submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        product_rating INTEGER DEFAULT NULL CHECK(product_rating BETWEEN 1 AND 5),
        service_rating INTEGER DEFAULT NULL CHECK(service_rating BETWEEN 1 AND 5),
        product_photo TEXT DEFAULT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
""")

# Admin table (No changes)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS admins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
""")

conn.commit()
conn.close()
print("✅ Database initialized successfully.")
