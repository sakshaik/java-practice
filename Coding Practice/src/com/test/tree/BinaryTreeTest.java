package com.test.tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;

public class BinaryTreeTest {
	int dia = 0;
	static Node root;

	public static void main(String[] args) {
		Node root1 = new Node(3);
		root1.left = new Node(2);
		root1.right = new Node(4);
		root1.left.left = new Node(1);
		root1.right.right = new Node(6);
		root1.right.left = new Node(5);

		Node root2 = new Node(1);
		root2.left = new Node(3);
		root2.right = new Node(2);
		root2.left.left = new Node(4);
		root2.left.right = new Node(5);

		BinaryTreeTest test = new BinaryTreeTest();

		System.out.println(test.checkBST(root1));

		System.out.println(test.findDist(root1, 3, 5));
		System.out.println(test.findDist(root1, 2, 4));

		Node lca = test.lca(root1, 2, 3);
		ArrayList<Integer> lcaList = test.inOrder(lca);

		System.out.println(Arrays.toString(lcaList.toArray()));

		ArrayList<Integer> boundary = test.boundary(root1);
		System.out.println(Arrays.toString(boundary.toArray()));

		ArrayList<Integer> diagonalList = test.diagonal(root1);
		System.out.println(Arrays.toString(diagonalList.toArray()));

		ArrayList<Integer> bottomView = test.bottomView(root1);
		System.out.println(Arrays.toString(bottomView.toArray()));

		System.out.println(test.longestConsecutive(root1));

		System.out.println(BinaryTreeTest.treePathsSum(root1));

		System.out.println(test.maxLevelSum(root1));

		ArrayList<Integer> preOrder = BinaryTreeTest.preorder(root1);
		System.out.println(Arrays.toString(preOrder.toArray()));

		System.out.println(test.check(root1));

		System.out.println(test.getLevelDiff(root1));

		ArrayList<Integer> noSibling = test.noSibling(root1);
		System.out.println(Arrays.toString(noSibling.toArray()));

		System.out.println(test.hasPathSum(root1, 8));
		System.out.println(test.hasPathSum(root1, 3));
		System.out.println(test.hasPathSum(root1, 5));

		System.out.println(test.isIdentical(root1, root2));

		ArrayList<Integer> inOrder = test.inOrder(root1);
		System.out.println(Arrays.toString(inOrder.toArray()));

		BinaryTreeTest.reverseAlternate(root1);

		ArrayList<Integer> inOrder2 = test.inOrder(root1);
		System.out.println(Arrays.toString(inOrder2.toArray()));

		ArrayList<Integer> spiralData = test.findSpiral(root1);
		System.out.println(Arrays.toString(spiralData.toArray()));

		System.out.println(test.isSumTree(root1));
		ArrayList<Integer> leftView = test.leftView(root1);
		System.out.println(Arrays.toString(leftView.toArray()));

		System.out.println(test.diameter(root1));

		ArrayList<Integer> kDist = test.Kdistance(root1, 0);
		System.out.println(Arrays.toString(kDist.toArray()));

		ArrayList<Integer> rightView = test.rightView(root1);
		System.out.println(Arrays.toString(rightView.toArray()));

		ArrayList<Integer> zigzag = test.zigZagTraversal(root1);
		System.out.println(Arrays.toString(zigzag.toArray()));

		System.out.println(test.leftLeavesSum(root1));
		postOrder(BinaryTreeTest.RemoveHalfNodes(root1));

		ArrayList<Integer> disagonalSum = BinaryTreeTest.diagonalSum(root1);
		System.out.println(Arrays.toString(disagonalSum.toArray()));

		System.out.println(BinaryTreeTest.isSumProperty(root1));

		System.out.println(BinaryTreeTest.maxNodeLevel(root1));

		System.out.println(test.isIsomorphic(root1, root2));

		ArrayList<Integer> inOrder3 = test.inOrder(root1);
		System.out.println(Arrays.toString(inOrder3.toArray()));

		ArrayList<Integer> verticalSums = test.verticalSum(root1);
		System.out.println(Arrays.toString(verticalSums.toArray()));

		System.out.println(test.height(root1));

		System.out.println(test.isFullTree(root1));

		System.out.println(BinaryTreeTest.isSymmetric(root1));

		int[] in = { 1, 6, 8, 7 };
		int[] pre = { 1, 6, 7, 8 };
		postOrder(BinaryTreeTest.buildTree(in, pre, in.length));

	}

	public static void postOrder(Node root) {
		if (root == null)
			return;
		postOrder(root.left);
		postOrder(root.right);
		System.out.print(root.data + " ");
	}

	ArrayList<Integer> inOrder(Node root) {
		ArrayList<Integer> inOrder = new ArrayList<Integer>();
		Node current, prev;
		if (root == null)
			return inOrder;
		current = root;
		while (current != null) {
			if (current.left == null) {
				inOrder.add(current.data);
				current = current.right;
			} else {
				prev = current.left;
				while (prev.right != null && prev.right != current)
					prev = prev.right;
				if (prev.right == null) {
					prev.right = current;
					current = current.left;
				} else {
					prev.right = null;
					inOrder.add(current.data);
					current = current.right;
				}
			}
		}
		return inOrder;
	}

	int height(Node node) {
		int height = 0;
		if (node == null) {
			return height;
		} else {
			int left = height(node.left);
			int right = height(node.right);
			if (left > right) {
				height = left + 1;
			} else {
				height = right + 1;
			}
		}
		return height;
	}

	boolean isFullTree(Node node) {
		if (node == null || (node.left == null && node.right == null)) {
			return true;
		}
		if (node.left != null && node.right != null) {
			return (isFullTree(node.left) && isFullTree(node.right));
		}
		return false;
	}

	public static boolean isSymmetric(Node root) {
		return isMirror(root, root);
	}

	private static boolean isMirror(Node head1, Node head2) {
		if (head1 == null && head2 == null) {
			return true;
		}
		if (head1 != null && head2 != null && head1.data == head2.data) {
			return (isMirror(head1.left, head2.right) && isMirror(head1.right, head2.left));
		}
		return false;
	}

	public static Node buildTree(int inorder[], int preorder[], int n) {
		Set<Node> set = new HashSet<>();
		Stack<Node> stack = new Stack<>();
		Node root = null;
		for (int pre = 0, in = 0; pre < preorder.length;) {
			Node node = null;
			do {
				node = new Node(preorder[pre]);
				if (root == null) {
					root = node;
				}
				if (!stack.isEmpty()) {
					if (set.contains(stack.peek())) {
						set.remove(stack.peek());
						stack.pop().right = node;
					} else {
						stack.peek().left = node;
					}
				}
				stack.push(node);
			} while (preorder[pre++] != inorder[in] && pre < preorder.length);
			node = null;
			while (!stack.isEmpty() && in < inorder.length && stack.peek().data == inorder[in]) {
				node = stack.pop();
				in++;
			}

			if (node != null) {
				set.add(node);
				stack.push(node);
			}
		}
		return root;
	}

	public ArrayList<Integer> verticalSum(Node root) {
		ArrayList<Integer> verticalSum = new ArrayList<Integer>();
		if (root == null) {
			return verticalSum;
		}
		TreeMap<Integer, Integer> hM = new TreeMap<Integer, Integer>();
		verticalSumUtil(root, 0, hM);
		Set set = hM.entrySet();
		Iterator it = set.iterator();
		while (it.hasNext()) {
			Map.Entry me = (Map.Entry) it.next();
			verticalSum.add((Integer) me.getValue());
		}
		return verticalSum;
	}

	public void verticalSumUtil(Node root, int hD, TreeMap<Integer, Integer> hM) {
		if (root == null) {
			return;
		}
		verticalSumUtil(root.left, hD - 1, hM);
		int prevSum = (hM.get(hD) == null) ? 0 : hM.get(hD);
		hM.put(hD, prevSum + root.data);
		verticalSumUtil(root.right, hD + 1, hM);
	}

	boolean isIsomorphic(Node root1, Node root2) {
		if (root1 == null && root2 == null) {
			return true;
		}
		if (root1 == null || root2 == null || root1.data != root2.data) {
			return false;
		}
		return ((isIsomorphic(root1.left, root2.left) && isIsomorphic(root1.right, root2.right))
				|| (isIsomorphic(root1.left, root2.right) && isIsomorphic(root1.right, root2.left)));
	}

	public static int maxNodeLevel(Node root) {
		if (root == null)
			return -1;
		Queue<Node> q = new LinkedList<Node>();
		q.add(root);
		int level = 0;
		int max = Integer.MIN_VALUE;
		int level_no = 0;

		while (true) {
			int NodeCount = q.size();
			if (NodeCount == 0)
				break;
			if (NodeCount > max) {
				max = NodeCount;
				level_no = level;
			}
			while (NodeCount > 0) {
				Node Node = q.peek();
				q.remove();
				if (Node.left != null)
					q.add(Node.left);
				if (Node.right != null)
					q.add(Node.right);
				NodeCount--;
			}
			level++;
		}
		return level_no;
	}

	public static int isSumProperty(Node root) {
		if (root == null || (root.left == null && root.right == null)) {
			return 1;
		} else {
			int left_data = root.left != null ? root.left.data : 0;
			int right_data = root.right != null ? root.right.data : 0;
			if ((root.data == (left_data + right_data)) && isSumProperty(root.left) != 0
					&& isSumProperty(root.right) != 0) {
				return 1;
			}
		}
		return 0;
	}

	public static ArrayList<Integer> diagonalSum(Node root) {
		ArrayList<Integer> arr = new ArrayList<>();
		getSum(root, arr, 0);
		return arr;
	}

	public static void getSum(Node root, ArrayList<Integer> arr, int x) {
		if (root == null)
			return;

		if (x == arr.size()) {
			arr.add(root.data);
		} else {
			int existing = arr.remove(x);
			arr.add(x, existing + root.data);
		}
		getSum(root.left, arr, x + 1);
		getSum(root.right, arr, x);
	}

	public static Node RemoveHalfNodes(Node root) {
		if (root == null) {
			return root;
		}
		root.left = RemoveHalfNodes(root.left);
		root.right = RemoveHalfNodes(root.right);

		if (root.left == null && root.right == null) {
			return root;
		} else if (root.left == null) {
			Node new_root = root.right;
			return new_root;
		} else if (root.right == null) {
			Node new_root = root.left;
			return new_root;
		}
		return root;
	}

	public static int findTreeHeight(Node root) {
		if (root == null)
			return 0;
		if (isLeaf(root))
			return 1;
		return 1 + Math.max(findTreeHeight(root.left), findTreeHeight(root.right));
	}

	static boolean isLeaf(Node node) {
		return (node.left != null && node.left.right == node && node.right != null && node.right.left == node);
	}

	public int leftLeavesSum(Node root) {
		int result = 0;
		if (root == null) {
			return 0;
		} else {
			if (isLeafNode(root.left)) {
				result += root.left.data;
			} else {
				result += leftLeavesSum(root.left);
			}
			result += leftLeavesSum(root.right);
		}
		return result;
	}

	private boolean isLeafNode(Node root) {
		if (root == null) {
			return false;
		} else if (root.left == null && root.right == null) {
			return true;
		}
		return false;
	}

	ArrayList<Integer> zigZagTraversal(Node root) {
		ArrayList<Integer> elements = new ArrayList<Integer>();
		if (root == null) {
			return elements;
		}

		Stack<Node> currentLevel = new Stack<>();
		Stack<Node> nextLevel = new Stack<>();

		currentLevel.push(root);
		boolean leftToRight = true;

		while (!currentLevel.isEmpty()) {
			Node node = currentLevel.pop();
			elements.add(node.data);
			if (leftToRight) {
				if (node.left != null) {
					nextLevel.push(node.left);
				}

				if (node.right != null) {
					nextLevel.push(node.right);
				}
			} else {
				if (node.right != null) {
					nextLevel.push(node.right);
				}

				if (node.left != null) {
					nextLevel.push(node.left);
				}
			}

			if (currentLevel.isEmpty()) {
				leftToRight = !leftToRight;
				Stack<Node> temp = currentLevel;
				currentLevel = nextLevel;
				nextLevel = temp;
			}
		}
		return elements;
	}

	ArrayList<Integer> rightView(Node node) {
		ArrayList<Integer> rightView = new ArrayList<Integer>();
		if (node == null) {
			return rightView;
		}
		Queue<Node> q = new LinkedList<Node>();
		q.add(node);
		while (!q.isEmpty()) {
			int n = q.size();
			for (int i = 0; i < n; i++) {
				Node current = q.peek();
				q.remove();
				if (i == n - 1) {
					rightView.add(current.data);
				}
				if (current.left != null) {
					q.add(current.left);
				}
				if (current.right != null) {
					q.add(current.right);
				}
			}
		}
		return rightView;
	}

	ArrayList<Integer> leftView(Node root) {
		ArrayList<Integer> leftView = new ArrayList<Integer>();
		if (root == null) {
			return leftView;
		}
		Queue<Node> q = new LinkedList<Node>();
		q.add(root);
		while (!q.isEmpty()) {
			int n = q.size();
			for (int i = 1; i <= n; i++) {
				Node current = q.poll();
				if (i == 1) {
					leftView.add(current.data);
				}
				if (current.left != null) {
					q.add(current.left);
				}
				if (current.right != null) {
					q.add(current.right);
				}
			}
		}
		return leftView;
	}

	boolean isSumTree(Node root) {
		if (root == null || isLeafNode(root)) {
			return true;
		}
		int ls, rs;
		if (isSumTree(root.left) && isSumTree(root.right)) {
			if (root.left == null)
				ls = 0;
			else if (isLeafNode(root.left))
				ls = root.left.data;
			else
				ls = 2 * (root.left.data);

			if (root.right == null)
				rs = 0;
			else if (isLeafNode(root.right))
				rs = root.right.data;
			else
				rs = 2 * (root.right.data);

			if ((root.data == rs + ls))
				return true;
			else
				return false;
		}
		return false;
	}

	ArrayList<Integer> Kdistance(Node root, int k) {
		ArrayList<Integer> kDist = new ArrayList<Integer>();

		kDistanceUtil(root, k, kDist);

		return kDist;
	}

	private void kDistanceUtil(Node root, int k, ArrayList<Integer> kdist) {
		if (root == null || k < 0) {
			return;
		} else if (k == 0) {
			kdist.add(root.data);
		}
		kDistanceUtil(root.left, k - 1, kdist);
		kDistanceUtil(root.right, k - 1, kdist);
	}

	void mirror(Node node) {
		node = mirrorUtil(node);
	}

	private Node mirrorUtil(Node node) {
		if (node == null) {
			return node;
		}
		Node left = mirrorUtil(node.left);
		Node right = mirrorUtil(node.right);
		node.left = right;
		node.right = left;
		return node;
	}

	boolean areMirror(Node a, Node b) {
		if (a == null && b == null) {
			return true;
		} else if (a == null || b == null) {
			return false;
		}
		return ((a.data == b.data) && areMirror(a.left, b.right) && areMirror(a.right, b.left));
	}

	int getMaxWidth(Node root) {
		int max = 0;
		if (root == null) {
			return max;
		}
		Queue<Node> q = new LinkedList<Node>();
		q.add(root);
		while (!q.isEmpty()) {
			int count = q.size();
			max = Math.max(max, count);
			while (count-- > 0) {
				Node temp = q.remove();
				if (temp.left != null) {
					q.add(temp.left);
				}
				if (temp.right != null) {
					q.add(temp.right);
				}
			}
		}
		return max;
	}

	int diameter(Node root) {
		diameterUtil(root);
		return dia;
	}

	public int diameterUtil(Node root) {
		if (root == null)
			return 0;
		int l = diameterUtil(root.left);
		int r = diameterUtil(root.right);
		if (l + r + 1 > dia)
			dia = l + r + 1;
		return 1 + Math.max(l, r);
	}

	public void toSumTree(Node root) {
		toSumTreeUtil(root);
	}

	private int toSumTreeUtil(Node root) {
		if (root == null) {
			return 0;
		}

		int old_val = root.data;

		root.data = toSumTreeUtil(root.left) + toSumTreeUtil(root.right);

		return root.data + old_val;
	}

	ArrayList<Integer> findSpiral(Node root) {
		ArrayList<Integer> spiralData = new ArrayList<Integer>();
		if (root == null) {
			return spiralData;
		}
		Stack<Node> s1 = new Stack<Node>();
		Stack<Node> s2 = new Stack<Node>();
		s1.push(root);
		while (!s1.empty() || !s2.empty()) {
			while (!s1.empty()) {
				Node temp = s1.peek();
				s1.pop();
				spiralData.add(temp.data);
				if (temp.right != null) {
					s2.push(temp.right);
				}
				if (temp.left != null) {
					s2.push(temp.left);
				}
			}

			while (!s2.empty()) {
				Node temp = s2.peek();
				s2.pop();
				spiralData.add(temp.data);
				if (temp.left != null) {
					s1.push(temp.left);
				}
				if (temp.right != null) {
					s1.push(temp.right);
				}
			}
		}
		return spiralData;
	}

	public ArrayList<Integer> reverseLevelOrder(Node node) {
		ArrayList<Integer> reverseOrder = new ArrayList<Integer>();
		if (node == null) {
			return reverseOrder;
		}

		Queue<Node> q = new LinkedList<Node>();
		Stack<Node> s = new Stack<Node>();
		q.add(node);

		while (!q.isEmpty()) {
			Node temp = q.peek();
			q.remove();
			s.push(temp);
			if (temp.right != null) {
				q.add(temp.right);
			}

			if (temp.left != null) {
				q.add(temp.left);
			}
		}

		while (!s.empty()) {
			Node temp = s.peek();
			s.pop();
			reverseOrder.add(temp.data);
		}
		return reverseOrder;
	}

	static void reverseAlternate(Node root) {
		if (root == null || (root.left == null && root.right == null)) {
			return;
		}
		reverseAlternateUtil(root.left, root.right, 0);
	}

	private static void reverseAlternateUtil(Node root1, Node root2, int level) {
		if (root1 == null || root2 == null) {
			return;
		}
		if (level % 2 == 0) {
			int temp = root1.data;
			root1.data = root2.data;
			root2.data = temp;
		}
		reverseAlternateUtil(root1.left, root2.right, level + 1);
		reverseAlternateUtil(root1.right, root2.left, level + 1);
	}

	boolean isIdentical(Node root1, Node root2) {
		if ((root1 == null && root2 == null)) {
			return true;
		} else if (root1 != null && root2 != null) {
			return ((root1.data == root2.data) && isIdentical(root1.left, root2.left)
					&& isIdentical(root1.right, root2.right));
		}
		return false;
	}

	boolean hasPathSum(Node root, int S) {
		boolean result = false;

		int deduct = S - root.data;
		if (deduct == 0 && root.left == null && root.right == null) {
			return (result = true);
		}
		if (root.left != null) {
			result = result || hasPathSum(root.left, deduct);
		}

		if (root.right != null) {
			result = result || hasPathSum(root.right, deduct);
		}

		return result;
	}

	ArrayList<Integer> noSibling(Node node) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		if (node == null) {
			result.add(-1);
			return result;
		}

		Queue<Node> q = new LinkedList<Node>();
		q.add(node);

		while (!q.isEmpty()) {
			Node temp = q.peek();
			q.remove();
			if (temp.left != null && temp.right == null) {
				result.add(temp.left.data);
			}
			if (temp.left == null && temp.right != null) {
				result.add(temp.right.data);
			}
			if (temp.left != null) {
				q.add(temp.left);
			}
			if (temp.right != null) {
				q.add(temp.right);
			}
		}

		if (result.isEmpty()) {
			result.add(-1);
		}

		Collections.sort(result);
		return result;
	}

	static ArrayList<ArrayList<Integer>> levelOrder(Node node) {
		Queue<Node> q = new LinkedList<>();
		q.add(node);
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		while (true) {
			int size = q.size();
			if (size == 0)
				break;
			ArrayList<Integer> level = new ArrayList<Integer>();
			while (size > 0) {
				node = q.peek();
				q.poll();
				level.add(node.data);
				if (node.left != null)
					q.add(node.left);
				if (node.right != null)
					q.add(node.right);
				size--;
			}
			result.add(level);
		}
		return result;
	}

	int minDepth(Node root) {
		if (root == null) {
			return 0;
		} else if (root.left == null && root.right == null) {
			return 1;
		}
		if (root.left == null) {
			return minDepth(root.right) + 1;
		}
		if (root.right == null) {
			return minDepth(root.left) + 1;
		}
		return Math.min(minDepth(root.left), minDepth(root.right)) + 1;
	}

	int getLevelDiff(Node root) {
		int evenSum = 0, oddSum = 0;
		boolean even = false;
		Queue<Node> q = new LinkedList<>();
		q.add(root);
		while (true) {
			int size = q.size();
			if (size == 0)
				break;
			while (size > 0) {
				Node temp = q.peek();
				q.poll();
				if (even) {
					evenSum += temp.data;
				} else {
					oddSum += temp.data;
				}
				if (temp.left != null)
					q.add(temp.left);
				if (temp.right != null)
					q.add(temp.right);
				size--;
			}
			even = !even;
		}
		return oddSum - evenSum;
	}

	boolean check(Node root) {
		if (root == null) {
			return true;
		}
		Queue<Node> q = new LinkedList<Node>();
		q.add(root);
		int check = Integer.MAX_VALUE;
		int level = 0;
		while (!q.isEmpty()) {
			level++;
			int size = q.size();
			while (size > 0) {
				Node temp = q.remove();
				if (temp.left != null) {
					q.add(temp.left);
					if (temp.left.left == null && temp.left.right == null) {
						if (check == Integer.MAX_VALUE) {
							check = level;
						} else if (check != level) {
							return false;
						}
					}
				}

				if (temp.right != null) {
					q.add(temp.right);
					if (temp.right.left == null && temp.right.right == null) {
						if (check == Integer.MAX_VALUE) {
							check = level;
						} else if (check != level) {
							return false;
						}
					}
				}

				size--;
			}
		}
		return true;
	}

	static ArrayList<Integer> preorder(Node root) {
		ArrayList<Integer> preOrder = new ArrayList<Integer>();
		preOrderUtil(root, preOrder);
		return preOrder;
	}

	private static void preOrderUtil(Node root, ArrayList<Integer> preOrder) {
		if (root != null) {
			preOrder.add(root.data);
			preOrderUtil(root.left, preOrder);
			preOrderUtil(root.right, preOrder);
		}
	}

	public int maxLevelSum(Node root) {
		int maxSum = Integer.MIN_VALUE;
		if (root == null) {
			return 0;
		}

		Queue<Node> q = new LinkedList<>();
		q.add(root);

		while (true) {
			int size = q.size();
			int levelSum = 0;
			if (size == 0)
				break;
			while (size > 0) {
				Node temp = q.peek();
				levelSum += temp.data;
				q.poll();
				if (temp.left != null)
					q.add(temp.left);
				if (temp.right != null)
					q.add(temp.right);
				size--;
			}
			if (levelSum > maxSum) {
				maxSum = levelSum;
			}
		}
		return maxSum;
	}

	public static long treePathsSum(Node root) {
		return treePathsSumUtil(root, 0);
	}

	private static long treePathsSumUtil(Node root, long data) {
		if (root == null) {
			return 0;
		}
		data = data * 10 + root.data;
		if (root.left == null && root.right == null) {
			return data;
		}
		return treePathsSumUtil(root.left, data) + treePathsSumUtil(root.right, data);
	}

	public int longestConsecutive(Node root) {
		if (root == null) {
			return 0;
		}
		int length = longestConsecutiveUtil(root, root.data - 1, 0);
		return length == 1 ? -1 : length;
	}

	private int longestConsecutiveUtil(Node root, int prev_val, int prev_len) {
		if (root == null) {
			return prev_len;
		}

		int curr_val = root.data;
		if (curr_val == prev_val + 1) {
			return Math.max(longestConsecutiveUtil(root.left, curr_val, prev_len + 1),
					longestConsecutiveUtil(root.right, curr_val, prev_len + 1));
		}

		int new_len = Math.max(longestConsecutiveUtil(root.left, curr_val, 1),
				longestConsecutiveUtil(root.right, curr_val, 1));
		return Math.max(new_len, prev_len);
	}

	Node buildTreeInAndPost(int in[], int post[], int n) {
		Stack<Node> st = new Stack<>();
		HashSet<Node> s = new HashSet<>();
		Node root = null;
		for (int p = n - 1, i = n - 1; p >= 0;) {
			Node node = null;
			do {
				node = new Node(post[p]);
				if (root == null) {
					root = node;
				}
				if (st.size() > 0) {
					if (s.contains(st.peek())) {
						s.remove(st.peek());
						st.peek().left = node;
						st.pop();
					} else {
						st.peek().right = node;
					}
				}
				st.push(node);
			} while (post[p--] != in[i] && p >= 0);
			node = null;
			while (st.size() > 0 && i >= 0 && st.peek().data == in[i]) {
				node = st.peek();
				st.pop();
				i--;
			}
			if (node != null) {
				s.add(node);
				st.push(node);
			}
		}
		return root;
	}

	public ArrayList<Integer> bottomView(Node root) {
		ArrayList<Integer> bottomView = new ArrayList<Integer>();
		TreeMap<Integer, int[]> m = new TreeMap<>();
		printBottomViewUtil(root, 0, 0, m);
		for (int val[] : m.values()) {
			bottomView.add(val[0]);
		}
		return bottomView;
	}

	private void printBottomViewUtil(Node root, int curr, int hd, TreeMap<Integer, int[]> m) {
		if (root == null)
			return;

		if (!m.containsKey(hd)) {
			m.put(hd, new int[] { root.data, curr });
		} else {
			int[] p = m.get(hd);
			if (p[1] <= curr) {
				p[1] = curr;
				p[0] = root.data;
			}
			m.put(hd, p);
		}
		printBottomViewUtil(root.left, curr + 1, hd - 1, m);
		printBottomViewUtil(root.right, curr + 1, hd + 1, m);
	}

	boolean isBalanced(Node root) {
		if (root == null) {
			return true;
		}
		int lh = height(root.left);
		int rh = height(root.right);
		if (Math.abs(lh - rh) <= 1 && isBalanced(root.left) && isBalanced(root.right)) {
			return true;
		}
		return false;
	}

	public static Node createTree(int parent[], int N) {
		Node[] created = new Node[N];
		for (int i = 0; i < N; i++)
			created[i] = null;

		for (int i = 0; i < N; i++)
			createNode(parent, i, created);

		return root;
	}

	static void createNode(int parent[], int i, Node created[]) {
		if (created[i] != null)
			return;

		created[i] = new Node(i);

		if (parent[i] == -1) {
			root = created[i];
			return;
		}

		if (created[parent[i]] == null)
			createNode(parent, parent[i], created);

		Node p = created[parent[i]];

		if (p.left == null)
			p.left = created[i];
		else
			p.right = created[i];
	}

	public void connect(Node root) {
		if (root == null) {
			return;
		}
		root.nextRight = null;
		while (root != null) {
			Node temp = root;
			while (temp != null) {
				if (temp.left != null) {
					if (temp.right != null) {
						temp.left.nextRight = temp.right;
					} else {
						temp.left.nextRight = nextRight(temp);
					}
				}
				if (temp.right != null) {
					temp.right.nextRight = nextRight(temp);
				}
				temp = temp.nextRight;
			}
			if (root.left != null) {
				root = root.left;
			} else if (root.right != null) {
				root = root.right;
			} else {
				root = nextRight(root);
			}
		}
	}

	private Node nextRight(Node temp) {
		Node a = temp.nextRight;
		while (a != null) {
			if (a.left != null)
				return a.left;
			if (a.right != null)
				return a.right;
			a = a.nextRight;
		}
		return null;
	}

	public ArrayList<Integer> diagonal(Node root) {
		ArrayList<Integer> diagonalList = new ArrayList<Integer>();
		if (root == null) {
			return diagonalList;
		}
		Node current = root;
		Queue<Node> q = new LinkedList<Node>();
		while (!q.isEmpty() || current != null) {
			if (current != null) {
				diagonalList.add(current.data);
				if (current.left != null) {
					q.add(current.left);
				}
				current = current.right;
			} else {
				current = q.remove();
			}
		}
		return diagonalList;
	}

	ArrayList<Integer> boundary;

	ArrayList<Integer> boundary(Node node) {
		boundary = new ArrayList<Integer>();
		if (node == null) {
			return boundary;
		}
		boundary.add(node.data);
		leftBoundary(node.left);
		leaves(node.left);
		leaves(node.right);
		rightBoundary(node.right);
		return boundary;
	}

	private void rightBoundary(Node node) {
		if (node == null) {
			return;
		}
		if (node.right != null) {
			rightBoundary(node.right);
			boundary.add(node.data);
		} else if (node.left != null) {
			rightBoundary(node.left);
			boundary.add(node.data);
		}
	}

	private void leaves(Node node) {
		if (node == null) {
			return;
		}
		leaves(node.left);
		if (node.left == null && node.right == null) {
			boundary.add(node.data);
		}
		leaves(node.right);
	}

	private void leftBoundary(Node node) {
		if (node == null) {
			return;
		}
		if (node.left != null) {
			boundary.add(node.data);
			leftBoundary(node.left);
		} else if (node.right != null) {
			boundary.add(node.data);
			leftBoundary(node.right);
		}
	}

	Node lca(Node root, int n1, int n2) {
		if (root == null) {
			return null;
		}

		if (root.data == n1 || root.data == n2) {
			return root;
		}

		Node leftLCA = lca(root.left, n1, n2);
		Node rightLCA = lca(root.right, n1, n2);

		if (leftLCA != null && rightLCA != null) {
			return root;
		}

		return leftLCA != null ? leftLCA : rightLCA;
	}

	public static int ans;

	int findDist(Node root, int a, int b) {
		ans = 0;
		findDistUtil(root, a, b);
		return ans;
	}

	private int findDistUtil(Node root, int a, int b) {
		if (root == null) {
			return 0;
		}
		int left = findDistUtil(root.left, a, b);
		int right = findDistUtil(root.right, a, b);
		if (root.data == a || root.data == b) {
			if (left != 0 || right != 0) {
				ans = Math.max(left, right);
				return 0;
			} else {
				return 1;
			}
		} else if (left != 0 && right != 0) {
			ans = left + right;
			return 0;
		} else if (left != 0 || right != 0) {
			return Math.max(left, right) + 1;
		}
		return 0;
	}

	boolean checkBST(Node root) {
		if (root == null || (root.left == null && root.right == null)) {
			return true;
		}
		if ((root.left != null && root.right == null && root.left.data < root.data)) {
			return checkBST(root.left);
		}

		if ((root.left == null && root.right != null && root.right.data > root.data)) {
			return checkBST(root.right);
		}

		if (root.data > root.left.data && root.data < root.right.data) {
			return checkBST(root.left) && checkBST(root.right);
		}

		return false;
	}

	// Huffman Decoding
	void decode(String s, Node root) {
		StringBuffer out = new StringBuffer();
		Node cur = root;
		for (char ch : s.toCharArray()) {
			if (ch == '0') {
				cur = cur.left;
				if (cur.left == null && cur.right == null && cur != null) {
					out.append(cur.data);
					cur = root;
				}
			}
			if (ch == '1') {
				cur = cur.right;
				if (cur.left == null && cur.right == null && cur != null) {
					out.append(cur.data);
					cur = root;
				}
			}

		}
		System.out.print(out.toString());
	}

	public Node sortedArrayToBST(int[] nums) {
		return build(nums, 0, nums.length - 1);
	}

	private Node build(int[] nums, int l, int r) {
		if (l > r) {
			return null;
		}
		int m = (l + r) / 2;
		Node node = new Node(nums[m]);
		node.left = build(nums, l, m - 1);
		node.right = build(nums, m + 1, r);
		return node;
	}

}

class Node {
	int data;
	int hd;
	Node nextRight;
	Node left;
	Node right;

	Node(int data) {
		this.data = data;
		hd = Integer.MAX_VALUE;
		left = null;
		right = null;
		nextRight = null;
	}
}