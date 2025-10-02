# MLOPS Healthcare Project step 1

In this folder I kept the data creation here for better organization.
When this starts you will create the data from the generator py for healthcare data

## Lets start!

## Heres the credentials you will need
## Database Connection



### DataGrip Setup:
1. **New Connection** → PostgreSQL
2. **Host:** localhost
3. **Port:** 5433  
4. **Database:** mlops_healthcare
5. **User:** mlops_user
6. **Password:** mlops_password
7. **Click "OK"** and allow introspection when prompted(This just refreshes if you dont already see the db)

### Python Connection:
```python

1. **Start the database:**

   docker-compose up -d