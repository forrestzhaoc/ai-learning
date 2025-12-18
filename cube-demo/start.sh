#!/bin/bash
export CUBEJS_DB_TYPE=sqlite
export CUBEJS_API_SECRET=secret
export CUBEJS_DEV_MODE=true

cd "$(dirname "$0")"
npm run dev

