cube('Orders', {
  sql_table: 'orders',
  
  measures: {
    count: {
      type: 'count',
    },
    
    totalAmount: {
      sql: 'amount',
      type: 'sum',
    },
    
    averageAmount: {
      sql: 'amount',
      type: 'avg',
    },
  },
  
  dimensions: {
    id: {
      sql: 'id',
      type: 'number',
      primary_key: true,
    },
    
    product: {
      sql: 'product',
      type: 'string',
    },
    
    createdAt: {
      sql: 'created_at',
      type: 'time',
    },
  },
});
