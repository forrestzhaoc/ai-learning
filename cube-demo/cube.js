module.exports = {
  dbType: 'sqlite',
  
  driverFactory: ({ dataSource }) => {
    const SqliteDriver = require('@cubejs-backend/sqlite-driver');
    return new SqliteDriver({ database: './cube.db' });
  },
};

