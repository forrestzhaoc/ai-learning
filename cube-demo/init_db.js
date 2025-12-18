const sqlite3 = require('sqlite3');
const db = new sqlite3.Database('./cube.db');

db.serialize(() => {
  db.run(`
    CREATE TABLE IF NOT EXISTS orders (
      id INTEGER PRIMARY KEY,
      product TEXT NOT NULL,
      amount INTEGER NOT NULL,
      created_at TEXT NOT NULL
    )
  `);

  db.run('DELETE FROM orders');

  const stmt = db.prepare('INSERT INTO orders (id, product, amount, created_at) VALUES (?, ?, ?, ?)');
  stmt.run(1, 'Product A', 100, '2024-01-01');
  stmt.run(2, 'Product B', 200, '2024-01-02');
  stmt.run(3, 'Product A', 150, '2024-01-03');
  stmt.run(4, 'Product C', 300, '2024-01-04');
  stmt.run(5, 'Product B', 250, '2024-01-05');
  stmt.finalize();

  console.log('✅ Database initialized successfully!');
  console.log('📊 Inserted 5 orders');
});

db.close();

