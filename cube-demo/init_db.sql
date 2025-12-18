CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    product TEXT NOT NULL,
    amount INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

DELETE FROM orders;

INSERT INTO orders (id, product, amount, created_at) VALUES
    (1, 'Product A', 100, '2024-01-01'),
    (2, 'Product B', 200, '2024-01-02'),
    (3, 'Product A', 150, '2024-01-03'),
    (4, 'Product C', 300, '2024-01-04'),
    (5, 'Product B', 250, '2024-01-05');

