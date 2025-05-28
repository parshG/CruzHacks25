from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)
detected_items = []

TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Shopping Cart</title>
  <script>
    function updateCart() {
      fetch('/items')
        .then(response => response.json())
        .then(data => {
          const list = document.getElementById('item-list');
          list.innerHTML = '';
          data.items.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            list.appendChild(li);
          });
        });
    }

    setInterval(updateCart, 1000); // Update every second
    window.onload = updateCart;
  </script>
</head>
<body>
  <h1>Detected Items in Cart</h1>
  <ul id="item-list">
    <!-- Items will be inserted here -->
  </ul>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(TEMPLATE)

@app.route("/update", methods=["POST"])
def update():
    global detected_items
    detected_items = request.json.get("items", [])
    return {"status": "updated"}

@app.route("/items")
def items():
    return jsonify({"items": detected_items})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
