## Dynamic Programming
### 0/1 Knapsack
You are given N items whose weights and values are known, as well as a knapsack to carry these items. The knapsack cannot carry more than a certain maximum weight, known as its capacity.

You need to maximize the total value of the items in your knapsack, while ensuring that the sum of the weights of the selected items does not exceed the capacity of the knapsack.

```
public static int findMaxKnapsackProfit(int capacity, int [] weights, int [] values) {
    // Create a table to hold intermediate values
    int n = weights.length;
    int[][] dp = new int[n + 1][capacity + 1];
    for(int[] row:dp) {
        Arrays.fill(row, 0);
    }

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= capacity; j++) {
            // Check if the weight of the current item is less than the current capacity
            if (weights[i - 1] <= j) {
                dp[i][j] = Math.max(values[i - 1] + dp[i - 1][j - weights[i - 1]],
                        dp[i - 1][j]);
            }
            // We don't include the item if its weight is greater than the current capacity
            else {
                dp[i][j] = dp[i - 1][j];
            }
        }
    }

    return dp[n][capacity];
}
```
