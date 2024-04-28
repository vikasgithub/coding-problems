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

## Two Pointers

### Sum of Three Values
```
	public static boolean findSumOfThree(int nums[], int target) {
		Arrays.sort(nums);
		int low, high, triples;

		for (int i = 0; i < nums.length - 2; i++) {
			low = i + 1;
			high = nums.length - 1;

			while (low < high) {
				triples = nums[i] + nums[low] + nums[high];

				if (triples == target) {
					return true;
				}
				else if (triples < target) {
					low++;
				} 
				else {
					high--;
				}
			}
		}

		return false;
	}
```
### Remove nth node from the end of the list
```
 public static LinkedListNode removeNthLastNode(LinkedListNode head, int n) {
    LinkedListNode right = head;
    LinkedListNode left = head;

    for (int i = 0; i < n; i++) {
      right = right.next;
    }

    if (right == null) {
      return head.next;
    }

    while (right.next != null) {
      right = right.next;
      left = left.next;
    }

    left.next = left.next.next;

    return head;
  }
```

### Sort Colors
Given an array, colors, which contains a combination of the following three elements:
- 0 (representing red)
- 1 (representing white)
- 2 (representing blue)

Sort the array in place so that the elements of the same color are adjacent, with the colors in the order of red, white, and blue. To improve your problem-solving skills, do not utilize the built-in sort function.

```
class Solution {
    public void sortColors(int[] nums) {
        int i = -1, j = nums.length, k = 0;
        while (k < j) {
            if (nums[k] == 0) {
                swap(nums, ++i, k++);
            } else if (nums[k] == 2) {
                swap(nums, --j, k);
            } else {
                ++k;
            }
        }
    }

    private void swap(int[] nums, int i, int j) {
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }
}
```
### Reverse Words in a String
```
class Solution {
    // A function that reverses characters from startRev to endRev in place
    private static void strRev(char[] str, int startRev, int endRev) {
        while (startRev < endRev) {
            char temp = str[startRev];
            str[startRev] = str[endRev];
            str[endRev] = temp;
            startRev++;
            endRev--;
        }
    }

    public static String reverseWords(String sentence) {
        sentence = sentence.replaceAll("\\s+", " ").trim();

        char[] charArray = sentence.toCharArray();
        int strLen = charArray.length;

        strRev(charArray, 0, strLen - 1);

        for (int start = 0, end = 0; end <= strLen; ++end) {
            if (end == strLen || charArray[end] == ' ') {
                strRev(charArray, start, end - 1);
                start = end + 1;
            }
        }

        return new String(charArray);
    }

    // Driver code
    public static void main(String[] args) {
        List<String> stringsToReverse = Arrays.asList(
            "Hello World",
            "a   string   with   multiple   spaces",
            "Case Sensitive Test 1234",
            "a 1 b 2 c 3 d 4 e 5",
            "     trailing spaces",
            "case test interesting an is this"
        );

        for (int i = 0; i < stringsToReverse.size(); i++) {
            System.out.println((i + 1) + ".\tOriginal string: '" + stringsToReverse.get(i) + "'");
            System.out.println("\tReversed string: '" + reverseWords(stringsToReverse.get(i)) + "'");
            System.out.println(new String(new char[100]).replace('\0', '-'));
        }
    }
}
```
## Fast and slow pointers
### Happy Number
```
public boolean isHappy(int n) {
        Set<Integer> vis = new HashSet<>();
        //break if number is 1 or is it repeated
        while (n != 1 && !vis.contains(n)) {
            vis.add(n);
            int x = 0;
            while (n != 0) {
                x += (n % 10) * (n % 10);
                n /= 10;
            }
            n = x;
        }
        return n == 1;
    }
```
### Detect cycle in a LinkedList
```
    public static boolean detectCycle(LinkedListNode head) {
        if (head == null) {
            return false;
        }
    
        LinkedListNode slow = head;
        LinkedListNode fast = head;

        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;

            if (slow == fast) {
                return true;
            }
        }
        
        return false;
    }
```
### Middle of a linked list
```
    public static LinkedListNode middleNode(LinkedListNode head) {

        LinkedListNode slow = head;
        LinkedListNode fast = head;

        while (fast != null && fast.next != null) {
            slow = slow.next;	
            fast = fast.next.next; 
        }

        return slow;
    }
```
### Circular array loop
```
class Solution {
    private int n;
    private int[] nums;

    public boolean circularArrayLoop(int[] nums) {
        n = nums.length;
        this.nums = nums;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == 0) {
                continue;
            }
            int slow = i, fast = next(i);
            while (nums[slow] * nums[fast] > 0 && nums[slow] * nums[next(fast)] > 0) {
                if (slow == fast) {
                    if (slow != next(slow)) {
                        return true;
                    }
                    break;
                }
                slow = next(slow);
                fast = next(next(fast));
            }
            int j = i;
            while (nums[j] * nums[next(j)] > 0) {
                nums[j] = 0;
                j = next(j);
            }
        }
        return false;
    }

    private int next(int i) {
        return (i + nums[i] % n + n) % n;
    }
}
```
### Palindrome linked list
- Find the middle node
- Reverse linked list from middle to the end
- compare the two halves

```
class PalindromeList {
    public static boolean palindrome(LinkedListNode head) {
       
        LinkedListNode slow = head;
        LinkedListNode fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        
        LinkedListNode revertData = LinkedListReversal.reverseLinkedList(slow);
        boolean check = compareTwoHalves(head, revertData);
        LinkedListReversal.reverseLinkedList(revertData);
        
        if (check) {
            return true;
        }

        return false;

    }

    public static boolean compareTwoHalves(LinkedListNode firstHalf, LinkedListNode secondHalf) {
        while (firstHalf != null && secondHalf != null) {
            if (firstHalf.data != secondHalf.data) {
                return false;
            } else {
                firstHalf = firstHalf.next;
                secondHalf = secondHalf.next;
            }


        }
        return true;
    }
```
## Modified Binary Search
### Binary Search
```
public static int binarySearch(int[] nums, int target) {
        int low = 0;
        int high = nums.length - 1;

        while (low <= high) {
            int mid = low + (high - low) / 2;

            if (nums[mid] == target) {
                return mid;
            }
            else if (target < nums[mid]) {
                high = mid - 1;
            }
            else if (target > nums[mid]) {
                low = mid + 1;
            }
        }

        return -1;
    }
```
### Search in a rotated array
```
public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) return -1;
        int start = 0, end = nums.length - 1;
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] == target) return mid;
            if (nums[start] < nums[mid]) { //Land in 1st continous section
                if (nums[start] <= target && target <= nums[mid]) end = mid;
                else start = mid;
            } else { //Land in 2nd continous section
                if (nums[mid] <= target && target <= nums[end]) start = mid;
                else end = mid;
            }
        }
        if (nums[start] == target) return start;
        if (nums[end] == target) return end;
        
        return -1;
    }
```
### Find K closest items
You are given a sorted array of integers, nums, and two integers, target and k. Your task is to return k number of integers that are close to the target value, target. The integers in the output array should be in a sorted order.
#### Solution 1 
```
class Solution {
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        List<Integer> ans = Arrays.stream(arr)
                                .boxed()
                                .sorted((a, b) -> {
                                    int v = Math.abs(a - x) - Math.abs(b - x);
                                    return v == 0 ? a - b : v;
                                })
                                .collect(Collectors.toList());
        ans = ans.subList(0, k);
        Collections.sort(ans);
        return ans;
    }
}
```
#### Solution 2
````
    class Solution {
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        int l = 0, r = arr.length;
        while (r - l > k) {
            if (x - arr[l] <= arr[r - 1] - x) {
                --r;
            } else {
                ++l;
            }
        }
        List<Integer> ans = new ArrayList<>();
        for (int i = l; i < r; ++i) {
            ans.add(arr[i]);
        }
        return ans;
    }
}
````
### Single element in a sorted array
```
    public static int singleNonDuplicate(int[] nums) {

        // initilaize the left and right pointer
        int l = 0;
        int r = nums.length - 1;

        while (l < r) {
           
            // if mid is odd, decrement it to make it even
            int mid = l + (r - l) / 2;
            if (mid % 2 == 1) mid--;
            
            // if the elements at mid and mid + 1 are the same, then the single element must appear after the midpoint
            if (nums[mid] == nums[mid + 1]) {
                l = mid + 2;
            } 
            // otherwise, we must search for the single element before the midpoint
            else {
                r = mid;
            }
        }
        return nums[l];
    }
```

## Merge Intervals
### Merge Intervals
```
class Solution {
    public List<Interval> merge(List<Interval> intervals) {
        if (intervals == null || intervals.size() <= 1) {
            return intervals;
        }

        intervals.sort(Comparator.comparing(interval -> interval.start)); // O(nlogn)
        int i = 0;
        while(i < intervals.size() - 1) {
            Interval curr = intervals.get(i), next = intervals.get(i + 1);
            if (curr.end >= next.start) {
                curr.end = curr.end >= next.end ? curr.end : next.end;
                intervals.remove(i + 1);
                continue;
            }
            i++;
        }
        return intervals;
    }
}
```
### Interval Intersection
```
    public static int[][] intervalsIntersection(int[][] intervalLista, int[][] intervalListb) {
        List<int[]> intersections = new ArrayList<>(); 
        int i = 0, j = 0;

        while (i < intervalLista.length && j < intervalListb.length) {
            int start = Math.max(intervalLista[i][0], intervalListb[j][0]);
            int end = Math.min(intervalLista[i][1], intervalListb[j][1]);
            
            if (start <= end) 
                intersections.add(new int[]{start, end}); 
            if (intervalLista[i][1] < intervalListb[j][1])
                i += 1;
            else
                j += 1;
        }

        return intersections.toArray(new int[0][]);
    }
```
### Employee Free time (hard)
```
 public static List<Interval> employeeFreeTime(List<List<Interval>> schedule) {
        PriorityQueue<int[]> heap = new PriorityQueue<>((a, b) -> a[0] - b[0]);
        
        for (int i = 0; i < schedule.size(); i++) {
            List<Interval> employeeSchedule = schedule.get(i);
            Interval interval = employeeSchedule.get(0);
            heap.offer(new int[]{interval.start, i, 0});
        }
        
        List<Interval> result = new ArrayList<>();
        
        int previous = schedule.get(heap.peek()[1]).get(heap.peek()[2]).start;
     
        while (!heap.isEmpty()) {
            int[] tuple = heap.poll();
            int i = tuple[1];
            int j = tuple[2];
            
            Interval interval = schedule.get(i).get(j);
            
            if (interval.start > previous) {
                result.add(new Interval(previous, interval.start));
            }
            
            previous = Math.max(previous, interval.end);
        
            if (j + 1 < schedule.get(i).size()) {
                Interval nextInterval = schedule.get(i).get(j + 1);
                heap.offer(new int[]{nextInterval.start, i, j + 1});
            }
        }
        return result;
    }
```

## In-place reversal of a linked list
### Reverse linked list
- Create dummy = head
- always insert in front of dummy, and set dummy to new head
- move head = head.next
```
class Solution {
    public ListNode reverseList(ListNode head) {
        if (head == null) return head;
        
        ListNode dummy = null;
        while(head != null) {
            ListNode temp = head.next;
            head.next = dummy;
            dummy = head;
            head = temp;
        }
        return dummy;
    }
}
```
### Reverse node in k - groups
```
 class Solution {

    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || head.next == null) return head;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy, cur = head;
        for (int i = 1; cur != null; ++i) {
            if (i % k == 0) {
                prev = reverse(prev, cur.next); // prev not included, cur included
                cur = prev.next;
            } else {
                cur = cur.next;
            }
        }
        return dummy.next;
    }

    private ListNode reverse(ListNode prev, ListNode end) {
        ListNode cur = prev.next;
        while (cur.next != end) {
            // p 1 2 3 end
            // p 2 1 3 end
            // p 3 2 1 end   cur always 1
            ListNode temp = cur.next;
            cur.next = temp.next;
            temp.next = prev.next;
            prev.next = temp;
        }
        return cur;
    }
}
```
### Reorder list
```
public class Solution {
    private ListNode reverse(ListNode head) {
        ListNode reversedList = null;
        while (head != null) {
            ListNode temp = head.next;
            head.next = reversedList;
            reversedList = head;
            head = temp;
        }
        return reversedList;
    }

    private void merge(ListNode head1, ListNode head2) {
        ListNode dummy = new ListNode(0);
        int index = 0;
        while (head1 != null && head2 != null) {
            if (index % 2 == 0) {
                dummy.next = head1;
                head1 = head1.next;
            } else {
                dummy.next = head2;
                head2 = head2.next;
            }
            dummy = dummy.next;
            index += 1;
        }
        if (head1 != null) {
            dummy.next = head1;
        } else if (head2 != null) {
            dummy.next = head2;
        }
    }

    private ListNode findMiddle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
    
    
    public void reorderList(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }

        ListNode mid = findMiddle(head);
        ListNode tail = reverse(mid.next);
        mid.next = null;

        merge(head, tail);
    }
}
```
### Swap kth node
You are given the head of a linked list, and an integer k.

Return the head of the linked list after swapping the values of the kth node from the beginning and the kth node from the end (the list is 1-indexed)
```
class Solution {
    public ListNode swapNodes(ListNode head, int k) {
        ListNode fast = head;
        while (--k > 0) {
            fast = fast.next;
        }
        ListNode p = fast;
        ListNode slow = head;
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        ListNode q = slow;
        int t = p.val;
        p.val = q.val;
        q.val = t;
        return head;
    }
}
```
### Reverse nodes in even-length groups
```
class ReverseNodes {
	public static LinkedListNode reverseEvenLengthGroups(LinkedListNode head)
	{
		LinkedListNode prev = head;
		LinkedListNode node, reverse, currNext, curr, prevNext = null;
		int groupLen = 2;
		int numNodes = 0;
		while(prev.next!= null)
		{
			node = prev;
			numNodes = 0;
			for (int i = 0; i < groupLen; i ++)
			{
				if(node.next == null)
                    break;
				numNodes += 1;
				node=node.next;
			}

			if(numNodes % 2 != 0) {
			   prev = node; 
			} else {
				reverse = node.next;
				curr = prev.next;
                for(int j=0; j < numNodes;j++){
                    currNext = curr.next;
				    curr.next = reverse;
				    reverse = curr;
				    curr = currNext;
                }
                prevNext = prev.next;
				prev.next = node;
			    prev = prevNext;
			}
			groupLen += 1;
		}
	    return head;
	}
}
```
### Swap nodes in pairs
Thoughts:
- swap
- move 2 steps, then swap again.
- becareful node.next == null, that's the end of list. no swapping.
```
 public ListNode swapPairs(ListNode head) {
    	if (head == null) {
    		return head;
    	}
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		head = dummy;
    	while (head.next != null && head.next.next != null) {
    		ListNode n1 = head.next;
    		ListNode n2 = head.next.next;
    		
    		n1.next = n2.next;
    		n2.next = n1;
    		n1 = n2;

    		head = head.next.next;
    	}
    	return dummy.next;
    }
```
## Two Heaps
### Maximize capital
```
public static int maximumCapital(int c, int k, int[] capitals, int[] profits) {
        int n = capitals.length;
		int currentCapital = c;
        PriorityQueue<int[]> capitalMinHeap = new PriorityQueue<>((a, b) -> a[0] - b[0]);
        for (int i = 0; i < n; ++i) {
            capitalMinHeap.offer(new int[] {capitals[i], i});
        }
        PriorityQueue<int[]> profitsMaxHeap = new PriorityQueue<>((a, b) -> b[0] - a[0]);
		int i = 0;
        while (i < k) {
            while (!capitalMinHeap.isEmpty() && capitalMinHeap.peek()[0] <= currentCapital) {
				int[] j = capitalMinHeap.poll();
                profitsMaxHeap.offer(new int[]{profits[j[1]]});
            }
            if (profitsMaxHeap.isEmpty()) {
                break;
            }
			int x = profitsMaxHeap.poll()[0];
            currentCapital += x;
			i++;
        }
        return currentCapital;
    }
```
### Sliding window median
```
class Solution {
    PriorityQueue<Integer> minHeap = new PriorityQueue<>();
    PriorityQueue<Integer> maxHeap = new PriorityQueue<>(10, Collections.reverseOrder());
    public double[] medianSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k <= 0) {
            return null;
        }
        
        int n = nums.length;
        double[] rst = new double[n - k + 1];
        // Build median structure with window
        for (int i = 0; i < k - 1; i++) {
            addNum(nums[i]);
        }
        
        // Calculate
        for (int i = 0; i < n - k + 1; i++) {
            addNum(nums[i + k - 1]);
            rst[i] = findMedian();
            removeNum(nums[i]);
        }
        
        return rst;
    }
    
    private double findMedian() {
        if (maxHeap.size() == minHeap.size()) {
            return maxHeap.peek() / 2.0 + minHeap.peek() / 2.0; // be careful with Integer.MAX_VALUE, consider breaking or use long
        } else {
            return maxHeap.peek();
        }
    }
    
    private void addNum(int num) {
        if (maxHeap.isEmpty()) {
            maxHeap.add(num);
            return;
        }
        int currentMedian = maxHeap.peek();
        if (num <= currentMedian) {
            maxHeap.offer(num);
        } else {
            minHeap.offer(num);
        }
        balance();
    }
    
    private void removeNum(int num) {
        int currentMedian = maxHeap.peek();
        if (num <= currentMedian) {
            maxHeap.remove(num);
        } else {
            minHeap.remove(num);
        }
        balance();
    }
    
    // helper
    private void balance() {
        if (maxHeap.size() > minHeap.size() + 1) {
            minHeap.offer(maxHeap.poll());
        } else if (maxHeap.size() < minHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
    }
}
```
### Median of data stream
```
class MedianOfStream {

  PriorityQueue<Integer> maxHeapForSmallNum; 
  PriorityQueue<Integer> minHeapForLargeNum; 

  public MedianOfStream() {
    maxHeapForSmallNum = new PriorityQueue<>((a, b) -> b - a);
    minHeapForLargeNum = new PriorityQueue<>((a, b) -> a - b);
  }

  public void insertNum(int num) {
    if (maxHeapForSmallNum.isEmpty() || maxHeapForSmallNum.peek() >= num)
      maxHeapForSmallNum.add(num);
    else
      minHeapForLargeNum.add(num);

    if (maxHeapForSmallNum.size() > minHeapForLargeNum.size() + 1)
      minHeapForLargeNum.add(maxHeapForSmallNum.poll());
    else if (maxHeapForSmallNum.size() < minHeapForLargeNum.size())
      maxHeapForSmallNum.add(minHeapForLargeNum.poll());
  }

  public double findMedian() {
    if (maxHeapForSmallNum.size() == minHeapForLargeNum.size()) {
      return maxHeapForSmallNum.peek() / 2.0 + minHeapForLargeNum.peek() / 2.0;
    }
    return maxHeapForSmallNum.peek();
  }
}
```
## K-way Merge
### Merge sorted array
You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.
```
class Solution {
    public void merge(int[] A, int m, int[] B, int n) {
        // write your code here
        int index = m + n - 1;
        while (m > 0 && n > 0) {
            if (A[m - 1] > B[n - 1]) {
                A[index--] = A[--m];
            } else {
                A[index--] = B[--n];
            }
        }//While
        
        // Put remain of B
        while (n > 0) {
            A[index--] = B[--n];
        }
    }
}
```
### Kth Smallest Number in M Sorted Lists
```
public static int mThLargest(int[][] arr, int m) { 
   
        // Create a min heap. Every 
        // heap node has first element of an array 
        PriorityQueue<Pair> pq = new PriorityQueue<Pair>(); 
        for (int i = 0; i < arr.length; i++) { 
            pq.add(new Pair(arr[i][0], i, 0)); 
        } 
   
        // Now one by one get the minimum element 
        // from min heap and replace it with next 
        // element of its array 
        int count = 0; 
        int i=0;
        int j=0;
        while (count < m && !pq.isEmpty()) { 
            Pair curr = pq.poll(); 
   
            // i ==> Array Number 
            // j ==> Index in the array number 
            i = curr.arrayNumber; 
            j = curr.index; 
   
            // The next element belongs to same array as current. 
            if (j + 1 < arr[i].length) { 
                pq.add(new Pair(arr[i][j + 1], i, j + 1)); 
            } 
            count++; 
        } 
        return arr[i][j]; 
    } 
```
### Find K Pairs with Smallest Sums
You are given two integer arrays nums1 and nums2 sorted in non-decreasing order and an integer k.
Define a pair (u, v) which consists of one element from the first array and one element from the second array.

Return the k pairs (u1, v1), (u2, v2), ..., (uk, vk) with the smallest sums.
```
class Solution {
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        PriorityQueue<int[]> q = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        for (int i = 0; i < Math.min(nums1.length, k); ++i) {
            q.offer(new int[] {nums1[i] + nums2[0], i, 0});
        }
        List<List<Integer>> ans = new ArrayList<>();
        while (!q.isEmpty() && k > 0) {
            int[] element = q.poll();
            ans.add(Arrays.asList(nums1[e[1]], nums2[e[2]]));
            --k;
            if (e[2] + 1 < nums2.length) {
                q.offer(new int[] {nums1[e[1]] + nums2[e[2] + 1], e[1], e[2] + 1});
            }
        }
        return ans;
    }
}
```
### Merge k sorted lists
```
class Solution {
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        PriorityQueue<int[]> q = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        for (int i = 0; i < Math.min(nums1.length, k); ++i) {
            q.offer(new int[] {nums1[i] + nums2[0], i, 0});
        }
        List<List<Integer>> ans = new ArrayList<>();
        while (!q.isEmpty() && k > 0) {
            int[] e = q.poll();
            ans.add(Arrays.asList(nums1[e[1]], nums2[e[2]]));
            --k;
            if (e[2] + 1 < nums2.length) {
                q.offer(new int[] {nums1[e[1]] + nums2[e[2] + 1], e[1], e[2] + 1});
            }
        }
        return ans;
    }
}
```
### Kth smallest element in sorted matrix
```
public class Main { 
    public static int mThLargest(int[][] arr, int m) { 
   
        // Create a min heap. Every 
        // heap node has first element of an array 
        PriorityQueue<Pair> pq = new PriorityQueue<Pair>(); 
        for (int i = 0; i < arr.length; i++) { 
            pq.add(new Pair(arr[i][0], i, 0)); 
        } 
   
        // Now one by one get the minimum element 
        // from min heap and replace it with next 
        // element of its array 
        int count = 0; 
        int i=0;
        int j=0;
        while (count < m && !pq.isEmpty()) { 
            Pair curr = pq.poll(); 
   
            // i ==> Array Number 
            // j ==> Index in the array number 
            i = curr.arrayNumber; 
            j = curr.index; 
   
            // The next element belongs to same array as current. 
            if (j + 1 < arr[i].length) { 
                pq.add(new Pair(arr[i][j + 1], i, j + 1)); 
            } 
            count++; 
        } 
        return arr[i][j]; 
    } 
```

