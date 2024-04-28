## Subsets
### Subsets
Given an array of integers, find all possible sets
```
class Solution {
    public static List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        helper(res, new ArrayList<>(), 0, nums);
        return res;
    }

    public static void helper(List<List<Integer>> res, List<Integer> currentSubset, int currentIndex, int[] nums) {
        if (currentIndex >= nums.length) {
            res.add(currentSubset);
            return;
        }
        helper(res, new ArrayList<>(currentSubset), currentIndex + 1, nums);
        currentSubset.add(nums[currentIndex]);
        helper(res, currentSubset, currentIndex + 1, nums);
    }
}
```
### Permutations
```
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (nums == null || nums.length == 0) return result;
        
        dfs(result, new ArrayList<>(), nums);
        return result;
    }
    
    private void dfs(List<List<Integer>> result, List<Integer> list, int[] nums) {
        if (list.size() == nums.length) {
            result.add(new ArrayList<>(list));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (list.contains(nums[i])) continue;
            list.add(nums[i]);
            dfs(result, list, nums);
            list.remove(list.size() - 1);
        }
    }
}
```

## Greedy Techniques
### Jump Game I
```
public static boolean jumpGame(int[] nums) {
    int targetNumIndex = nums.length - 1;
    for (int i = nums.length - 2; i >= 0; i--) {
        if (targetNumIndex <= (i + nums[i])) {
            targetNumIndex = i;
        }
    }
    if (targetNumIndex == 0)
        return true;
    return false;
}
```
### Boats to save people
```
public static int rescueBoats(int[] people, int limit) {
    Arrays.sort(people);
    
    int left = 0; 
    int right = people.length - 1;

    int boats = 0; 

    while (left <= right) {
        if (people[left] + people[right] <= limit) {
            left++;
        }
        right--;

        boats++;
    }
    return boats;
}
```
### Gas Stations
