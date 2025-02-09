from flask import Flask, render_template
from flask_socketio import SocketIO
import random
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Mock function to simulate live stock data
def generate_stock_data():
    while True:
        stock_data = {
            'symbol': 'AAPL',
            'price': round(150 + random.uniform(-1, 1), 2),  # Random price
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')  # Current time
        }
        socketio.emit('stock_update', stock_data)  # Send data to clients
        time.sleep(2)  # Update every 2 seconds

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    # Start a background thread to generate stock data
    threading.Thread(target=generate_stock_data, daemon=True).start()
    socketio.run(app, debug=True)