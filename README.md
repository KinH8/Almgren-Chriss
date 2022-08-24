# Almgren-Chriss optimal execution
We demonstrate the use of Almgren-Chriss optimal execution model on a Malaysian large cap stock

We run the Almgren-Chriss model on a single stock (in this case a Malaysian large cap stock) to generate an optimal trading trajectory for the stock over a pre-specified period. The model is based on Almgren and Chriss's seminal 2000 paper Optimal Execution of Portfolio Transactions. For a given risk aversion parameter lambda, the model minimizes a combination of volatility risk and transaction costs arising from permanent and temporary market impact. Nowadays many extensions have been added to the base Almgren-Chriss model to allow for other features, i.e. self exciting price impact.

The trade trajectory depends on the choice of risk parameter. The figure below shows the trade path of 2 risk parameters. Lambda of 2E-6 is chosen by a risk averse trader who wishes to sell quickly to reduce exposure to volatility risk, despite the trading costs incurred in doing so. Conversely lambda of -2E-6 is chosen only a trader who likes risk. He postpones selling, thus incurring both higher expected trading costs due to his rapid sales at the end, and higher variance during the extended period that he holds the security,

![image](https://user-images.githubusercontent.com/105033135/186359619-7f3925bc-7f84-48ef-abc6-239c9c90277f.png)

Here is the efficient frontier of the strategy by risk parameter lambda. As we move to the right of the curve we experience a fall in expected loss as market impact cost goes down, at the expense of higher variance.

![image](https://user-images.githubusercontent.com/105033135/186371162-e125214e-5a89-40c9-b5dd-c31814ee80a9.png)


