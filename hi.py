import mysql.connector
mydb = mysql.connector.connect(
            host = "localhost",
            user = "root",
            passwd = "123Password",
            database = "Options"
        )

mycursor = mydb.cursor()

mycursor.execute("CREATE TABLE Options.'BlackScholesInputs' ('CalculationID' INT NOT NULL AUTO_INCREMENT, 'Stock Price' DECIMAL(18,9) NOT NULL, 'Strike Price' DECIMAL(18,9) NOT NULL, 'Interest rate' DECIMAL(18,9) NOT NULL, 'Volatility' DECIMAL(18,9) NOT NULL, 'TimeToExpiry' DECIMAL(18,9) NOT NULL, PRIMARY KEY ('CalculationID'));")