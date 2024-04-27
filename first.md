# Coding problem solutions

## Pattern: Sliding Window
### Sliding window maximum

```
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums == null || nums.length == 0) return new int[0];
        int[] result = new int[nums.length - k + 1];
        LinkedList<Integer> deque = new LinkedList<>();
        for(int i = 0; i < nums.length; i++){
            if(!deque.isEmpty() && deque.peek() == i - k) deque.poll();
            while(!deque.isEmpty() && nums[i] >= nums[deque.peekLast()]) deque.pollLast();
            deque.offer(i);
            if(i >= k - 1) result[i - k + 1] = nums[deque.peek()];
        }
        return result;
    }
}
```

* We can use Deque for this question. In deque, we only save the index of each number.
* For each index, we check if the peek is the one leaving the window, if so, remove it.
* Then we check from the end of the deque, remove the numbers that are smaller than the current one.
* Then we can add the number corresponding to peek index to our result for current position.

### Minimum window subsequence

```
class Solution {
public String minWindow(String s, String t) {

    int sIndex = 0, tIndex = 0, start = -1;
    int m = s.length(), n = t.length(), minLength = m;
    char[] ss = s.toCharArray(), tt = t.toCharArray();
  
    while (sIndex < m) {
        if(ss[sIndex] == tt[tIndex]) { // char match
            if(tIndex++ == n - 1) { // tIndex exhausted, process it
                int end = sIndex + 1; // mark end of candidate
                // reset tIndex to 0 and backtrack sIndex to 1st match position
                tIndex = 0;
                sIndex = backtrack(ss, tt, sIndex);

                // record the candidate
                if (end - sIndex < minLength) {
                    minLength = end - sIndex;
                    start = sIndex;
                }
            }
        }
        sIndex++;
    }
    return start == -1 ? "" : s.substring(start, start + minLength);
}

private int backtrack(char[] ss, char[] tt, int sIndex) {
    for (int i = tt.length - 1; i >= 0; i--) {
        while(ss[sIndex--] != tt[i]);
    }
    return ++sIndex; // sIndex = 1st char match index - 1; ++ to reset
}
```

### Longest repeating character
```
public class RepeatingCharacter {
    public static int longestRepeatingCharacterReplacement(String s, int k) {
        int stringLength = s.length();
        int lengthOfMaxSubstring = 0;
        int start = 0;
        Map<Character, Integer> charFreq = new HashMap<>();
        int mostFreqCharFrequency = 0;

        for (int end = 0; end < stringLength; ++end) {
            char currentChar = s.charAt(end);
            
            charFreq.put(currentChar, charFreq.getOrDefault(currentChar, 0) + 1);
            
            mostFreqCharFrequency = Math.max(mostFreqCharFrequency, charFreq.get(currentChar));

            //number of replacements = end - start + 1 - mostFreqCharFrequency
            if (end - start + 1 - mostFreqCharFrequency > k) {
                charFreq.put(s.charAt(start), charFreq.get(s.charAt(start)) - 1);
                start += 1;
            }

            lengthOfMaxSubstring = Math.max(lengthOfMaxSubstring, end - start + 1);
        }

        return lengthOfMaxSubstring;
    }
```

### Minimum Window Substring
```
    public static String minWindow(String s, String t) {
        if (t.equals("")) {
            return "";
        }

        Map<Character, Integer> reqCount = new HashMap<>();
        Map<Character, Integer> window = new HashMap<>();

        for (int i = 0; i < t.length(); i++) {
            char c = t.charAt(i);
            reqCount.put(c, 1 + reqCount.getOrDefault(c, 0));
        }

        for (int i = 0; i < t.length(); i++) {
            char c = t.charAt(i);
            window.put(c, 0);
        }

        int current = 0;
        int required = reqCount.size();

        int[] res = {-1, -1};
        int resLen = Integer.MAX_VALUE;

        int left = 0;
        for (int right = 0; right < s.length(); right++) {
            char c = s.charAt(right);

            if (t.indexOf(c) != -1) {
                window.put(c, 1 + window.getOrDefault(c, 0));
            }

            if (reqCount.containsKey(c) && window.get(c).equals(reqCount.get(c))) {
                current += 1;
            }

            while (current == required) {
                if ((right - left + 1) < resLen) {
                    res[0] = left;
                    res[1] = right;
                    resLen = (right - left + 1);
                }

                char leftChar = s.charAt(left);
                if (t.indexOf(leftChar) != -1) {
                    window.put(leftChar, window.get(leftChar) - 1);
                }

                if (reqCount.containsKey(leftChar) && window.get(leftChar) < reqCount.get(leftChar)) {
                    current -= 1;
                }
                left += 1;
            }
        }

        int leftIndex = res[0];
        int rightIndex = res[1];
        return resLen != Integer.MAX_VALUE ? s.substring(leftIndex, rightIndex + 1) : "";
    }
```

### Longest Substring without Repeating Characters
- 
```
class Solution {
    public int lengthOfLongestSubstring(String s) {
        boolean[] ss = new boolean[128];
        int ans = 0;
        for (int start = 0, end = 0; end < s.length(); ++end) {
            char c = s.charAt(end);
            while (ss[c]) {
                ss[s.charAt(start++)] = false;
            }
            ss[c] = true;
            ans = Math.max(ans, end - start + 1);
        }
        return ans;
    }
}
```

### Minimum Size Subarray Sum
Given an array of positive integers, nums, and a positive integer, target, find the minimum length of a contiguous subarray whose sum is greater than or equal to the target. If no such subarray is found, return 0.
```
public static int minSubArrayLen(int target, int[] nums) {
        int windowSize = Integer.MAX_VALUE;
        int currSubArrSize = 0;
        int start = 0;
        int sum = 0;

        for (int end = 0; end < nums.length; end++) {
            sum += nums[end];
            while (sum >= target) {
                currSubArrSize = (end + 1) - start;
                windowSize = Math.min(windowSize, currSubArrSize);
                sum -= nums[start];
                start += 1;
            }
        }

        if (windowSize != Integer.MAX_VALUE) {
            return windowSize;
        }
        return 0;
    }

```

### Best Time to Buy and Sell Stock
Given an array where the element at the index i represents the price of a stock on day i, find the maximum profit that you can gain by buying the stock once and then selling it.
```
class Solution {
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length <= 1) return 0;
        int profit = 0, min = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < min) min = prices[i];
            else profit = Math.max(profit, prices[i] - min);
        }
        return profit;
    }
}
```

## 