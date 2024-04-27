# Sliding window maximum

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

# Minimum window subsequence


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
