## Tree 
### Pre-Order Traversal in Binary Search Trees (M-L-R)
```
	public static void preTraverse(Node root) {

		if (root == null) return;

		System.out.print(root.getData() + ",");
    preTraverse(root.getLeftChild());
    preTraverse(root.getRightChild());

	}
```
### In-order Traversal in Binary Search Trees (L-M-R)
```
	public static void inTraverse(Node root) {

		if (root == null) return;

		inTraverse(root.getLeftChild());
		System.out.print(root.getData() + ",");
		inTraverse(root.getRightChild());

	}
```
### Post-Order Traversal in Binary Search Tree (L-R-M)
```
	public static void postTraverse(Node root) {

		if (root == null) return;

		postTraverse(root.getLeftChild());
		postTraverse(root.getRightChild());
		System.out.print(root.getData() + ",");

	}
```
### Find the Minimum Value in a Binary Search Tree
```
public static int findMin(Node root) {
        if (root == null)
            return -1;
        // In Binary Search Tree, all values in current node's left subtree are smaller 
        // than the current node's value.
        // So keep traversing (in order) towards left till you reach leaf node, and then return leaf node's value
        while (root.getLeftChild() != null) {
            root = root.getLeftChild();
        }
        return root.getData();
    }
```
###  Find kth Maximum Value in a Binary Search Tree
```
public static int findKthMax(Node root, int k) {

        //Perform In-Order Traversal to get sorted array. (ascending order)
        //Return value at index [length - k]
        StringBuilder result = new StringBuilder(); //StringBuilder is mutable
        result = inOrderTraversal(root, result);

        String[] array = result.toString().split(","); //Spliting String into array of strings
        if ((array.length - k) >= 0) return Integer.parseInt(array[array.length - k]);

        return -1;
    }

    //Helper recursive function to traverse tree using inorder traversal
    //and return result in StringBuilder
    public static StringBuilder inOrderTraversal(Node root, StringBuilder result) {

        if (root.getLeftChild() != null) inOrderTraversal(root.getLeftChild(), result);

        result.append(root.getData() + ",");

        if (root.getRightChild() != null) inOrderTraversal(root.getRightChild(), result);

        return result;
    }
```
### Flatten Binary Tree to Linked List
```
public static TreeNode<Integer> flattenTree(TreeNode<Integer> root) {
		if (root == null) {
			return null;
		}

		TreeNode<Integer> current = root;
		while (current != null) {
			
			if (current.left != null) {
				
				TreeNode<Integer> last = current.left;
				while (last.right != null) {
					last = last.right;
				}

				last.right = current.right;
				current.right = current.left;
				current.left = null;

			}
			current = current.right;
		}
		return root;
	}
```
### Diameter of Binary Tree
```
class Pair {
    int diameter;
    int height;

    public Pair(int diameter, int height) {
      this.diameter = diameter;
      this.height = height;
    }
}

public class Solution {

  // Helper function to calculate diameter and height of a binary tree
  public static Pair diameterHelper(TreeNode<Integer> node) {
    if (node == null) {
      // If the node is null, return Pair with diameter and height both as 0
      return new Pair(0, 0);
    } else {
      // Recursively calculate the Pair for left and right subtrees
      Pair lh = diameterHelper(node.left);
      Pair rh = diameterHelper(node.right);

      // Calculate height as the maximum height of left and right subtrees + 1
      int height = Math.max(lh.height, rh.height) + 1;

      // Calculate diameter as the maximum of left diameter, right diameter, and the sum of left and right heights
      int diameter = Math.max(lh.diameter, Math.max(rh.diameter, lh.height + rh.height));

      // Return the Pair for the current subtree
      return new Pair(diameter, height);
    }
  }

  // Function to find the diameter of a binary tree
  public static int diameterOfBinaryTree(TreeNode<Integer> root) {
    if (root == null)
      // If the root is null, return 0 as the diameter
      return 0;

    // Calculate the Pair for the entire tree using the helper function
    Pair pair = diameterHelper(root);

    // Return the diameter from the Pair
    return pair.diameter;
  }
}
```
### Serialize Deserialize tree
```
class SerializeDeserialize {
    // Initializing our marker as the max possible int value
    private static final String MARKER = "M";
    private static int m = 1;

    private static void serializeRec(TreeNode<Integer> node, List<String> stream) {
        // Adding marker to stream if the node is null
        if (node == null) {
            String s = Integer.toString(m);
            stream.add(MARKER+s);
            m = m + 1;
            return;
        }

        // Adding node to stream
        stream.add(String.valueOf(node.data));

        // Doing a pre-order tree traversal for serialization
        serializeRec(node.left, stream);
        serializeRec(node.right, stream);
    }

    // Function to serialize tree into a list.
    public static List<String> serialize(TreeNode<Integer> root) {
        List<String> stream = new ArrayList<>();
        serializeRec(root, stream);
        return stream;
    }

    public static TreeNode<Integer> deserializeHelper(List<String> stream) {
        // pop last element from list
        String val = stream.remove(stream.size()-1);

        // Return null when a marker is encountered
        if (val.charAt(0) == MARKER.charAt(0)) {
            return null;
        }

        // Creating new Binary Tree Node from current value from stream
        TreeNode<Integer> node = new TreeNode<Integer>(Integer.parseInt(val));

        // Doing a pre-order tree traversal for deserialization
        node.left = deserializeHelper(stream);
        node.right = deserializeHelper(stream);

        // Return node if it exists
        return node;
    }

    // Function to deserialize list into a binary tree.
    public static TreeNode<Integer> deserialize(List<String> stream){
        Collections.reverse(stream);
        TreeNode<Integer> node = deserializeHelper(stream);
        return node;
    }
}
```
### Invert binary tree
```
public void invertBinaryTree(TreeNode root) {
        if (root == null) {
            return;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()) {
            TreeNode node = queue.poll();
            TreeNode temp = node.left;
            node.left = node.right;
            node.right = temp;
            if (node.left != null) {
                queue.offer(node.left);
            }
            if (node.right != null) {
                queue.offer(node.right);
            }
        }
    }
```
### Binary Tree Maximum Path Sum
```
class Solution {
    private int ans = -1001;

    public int maxPathSum(TreeNode root) {
        dfs(root);
        return ans;
    }

    private int dfs(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = Math.max(0, dfs(root.left));
        int right = Math.max(0, dfs(root.right));
        ans = Math.max(ans, root.val + left + right);
        return root.val + Math.max(left, right);
    }
}
```
### Converted sorted array to binary tree
```
class SortedArrayToBST {
    public static TreeNode<Integer> sortedArrayToBST(int[] nums) {
        // Call the helper function
        return sortedArrayToBSTHelper(nums, 0, nums.length - 1);
    }

    private static TreeNode<Integer> sortedArrayToBSTHelper(int[] nums, int low, int high) {
        // Base case: if low > high, then there are no more elements to add to the BST
        if (low > high) {
            return null;
        }

        // Calculate the middle index
        int mid = low + (high - low) / 2;

        // Center value of sorted array as the root of the BST
        TreeNode<Integer> root = new TreeNode<Integer>(nums[mid]);

        // Recursively add the elements in nums[low:mid-1] to the left subtree of root
        root.left = sortedArrayToBSTHelper(nums, low, mid - 1);

        // Recursively add the elements in nums[mid+1:high] to the right subtree of root
        root.right = sortedArrayToBSTHelper(nums, mid + 1, high);

        // Return the root node
        return root;
    }
```
### Build binary tree from pre-order and in-order traversal (review)
```
class Solution {
    private int[] preorder;
    private Map<Integer, Integer> d = new HashMap<>();

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int n = preorder.length;
        this.preorder = preorder;
        for (int i = 0; i < n; ++i) {
            d.put(inorder[i], i);
        }
        return dfs(0, 0, n);
    }

    private TreeNode dfs(int i, int j, int n) {
        if (n <= 0) {
            return null;
        }
        int v = preorder[i];
        int k = d.get(v);
        TreeNode l = dfs(i + 1, j, k - j);
        TreeNode r = dfs(i + 1 + k - j, k + 1, n - 1 - (k - j));
        return new TreeNode(v, l, r);
    }
}
```
### Binary tree right side view
```
right side view:
- the tree may not be complete
- always find right-most. if right child not available, dfs into left child
- tracking back is hard for dfs
- bfs: on each level, record the last item of the queue
*/

class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;

        // init queue
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        // loop over queue with while loop; inner while loop to complete level
        while (!queue.isEmpty()) {
            int size = queue.size();
            while (size > 0) {
                size--;
                TreeNode node = queue.poll();
                if (size == 0) result.add(node.val);
                if(node.left != null) queue.offer(node.left);
                if(node.right != null) queue.offer(node.right);
            }
        }
        return result;
    }
}
```
### Lowest common ancenstor in a binary tree
```
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }
        var left = lowestCommonAncestor(root.left, p, q);
        var right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) {
            return root;
        }
        return left == null ? right : left;
    }
}
```
### Validate binary search tree
```
class Solution {
    public boolean isValidBST(TreeNode root) {
        return dfs(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }
    
    private boolean dfs(TreeNode node, long min, long max) {
        if (node == null) return true;
        if (node.val <= min || node.val >= max) return false;
        return dfs(node.left, min, node.val) && dfs(node.right, node.val, max);
    }
}
```
### Maximum depth of a binary tree
```
public class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }
}
```
### Kth Smallest Element in a BST
```
public int kthSmallest(TreeNode root, int k) {
        if (root == null) {
            return -1;
        }
        List<Integer> ans = new ArrayList<>();
        dfs(root, ans, k);

        return ans.get(ans.size() - 1);
    }

    private void dfs(TreeNode root, List<Integer> ans, int k) {
        if (root == null || ans.size() == k) {
            return;
        }
        if (root.left != null) {
            dfs(root.left, ans, k);
        }
        if (ans.size() == k) {
            return;
        }
        ans.add(root.val);
        if (root.right != null) {
            dfs(root.right, ans, k);
        }
    } 
```
## Tree Breadth first search