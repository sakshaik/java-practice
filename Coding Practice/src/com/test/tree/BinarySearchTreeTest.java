package com.test.tree;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Stack;

public class BinarySearchTreeTest {

	public static void main(String[] args) {
		BinaryTreeNode root = null;
		root = BinarySearchTreeTest.insert(root, 50);
		BinarySearchTreeTest.inorder(root);
		root = BinarySearchTreeTest.insert(root, 30);
		BinarySearchTreeTest.inorder(root);
		root = BinarySearchTreeTest.insert(root, 20);
		BinarySearchTreeTest.inorder(root);
		root = BinarySearchTreeTest.insert(root, 40);
		BinarySearchTreeTest.inorder(root);
		root = BinarySearchTreeTest.insert(root, 70);
		BinarySearchTreeTest.inorder(root);
		root = BinarySearchTreeTest.insert(root, 60);
		BinarySearchTreeTest.inorder(root);
		root = BinarySearchTreeTest.insert(root, 80);
		BinarySearchTreeTest.inorder(root);
		System.out.println(BinarySearchTreeTest.printNearNodes(root, 30, 70));
		BinarySearchTreeTest.inorder(BinarySearchTreeTest.LCA(root, 20, 50));

	}

	private static BinaryTreeNode insert(BinaryTreeNode root, int Key) {
		if (root == null) {
			root = new BinaryTreeNode(Key);
			return root;
		}
		if (Key < root.data) {
			root.left = insert(root.left, Key);
		} else if (Key > root.data) {
			root.right = insert(root.right, Key);
		}
		return root;
	}

	private static BinaryTreeNode LCA(BinaryTreeNode root, int n1, int n2) {
		if (root == null) {
			return null;
		}
		if (root.data > n1 && root.data > n2) {
			return LCA(root.left, n1, n2);
		}
		if (root.data < n1 && root.data < n2) {
			return LCA(root.right, n1, n2);
		}
		return root;
	}

	private static void inorder(BinaryTreeNode root) {
		if (root != null) {
			inorder(root.left);
			System.out.println(root.data);
			inorder(root.right);
		}
	}

	public static ArrayList<Integer> printNearNodes(BinaryTreeNode root, int low, int high) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		printUtil(root, low, high, result);
		return result;
	}

	private static void printUtil(BinaryTreeNode root, int low, int high, ArrayList<Integer> result) {
		if (root == null) {
			return;
		}
		if (low < root.data) {
			printUtil(root.left, low, high, result);
		}
		if (root.data >= low && root.data <= high) {
			result.add(root.data);
		}
		printUtil(root.right, low, high, result);
	}

	BinaryTreeNode binaryTreeToBST(BinaryTreeNode root) {
		List<Integer> list = new ArrayList<Integer>();
		preOrder(root, list);
		Collections.sort(list);
		int[] index = { 0 };
		inOrder(root, list, index);
		return root;
	}

	void preOrder(BinaryTreeNode root, List<Integer> list) {
		if (root == null) {
			return;
		}
		list.add(root.data);
		preOrder(root.left, list);
		preOrder(root.right, list);
	}

	void inOrder(BinaryTreeNode root, List<Integer> list, int[] index) {
		if (root == null) {
			return;
		}
		inOrder(root.left, list, index);
		root.data = list.get(index[0]);
		index[0]++;
		inOrder(root.right, list, index);
	}

	public int kthLargest(Node root, int K) {
		Node curr = root;
		Node kthLargest = null;
		int count = 0;
		while (curr != null) {
			if (curr.right == null) {
				if (++count == K) {
					kthLargest = curr;
				}
				curr = curr.left;
			} else {
				Node next = curr.right;
				while (next.left != null && next.left != curr) {
					next = next.left;
				}
				if (next.left == null) {
					next.left = curr;
					curr = curr.right;
				} else {
					next.left = null;
					if (++count == K) {
						kthLargest = curr;
					}
					curr = curr.left;
				}
			}
		}
		return kthLargest.data;
	}

	public static ArrayList<Integer> findCommon(BinaryTreeNode root1, BinaryTreeNode root2) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		Stack<BinaryTreeNode> s1 = new Stack<BinaryTreeNode>();
		Stack<BinaryTreeNode> s2 = new Stack<BinaryTreeNode>();
		while (true) {
			if (root1 != null) {
				s1.push(root1);
				root1 = root1.left;
			} else if (root2 != null) {
				s2.push(root2);
				root2 = root2.left;
			} else if (!s1.isEmpty() && !s2.isEmpty()) {
				root1 = s1.peek();
				root2 = s2.peek();
				if (root1.data == root2.data) {
					result.add(root1.data);
					s1.pop();
					s2.pop();
					root1 = root1.right;
					root2 = root2.right;
				} else if (root1.data < root2.data) {
					s1.pop();
					root1 = root1.right;
					root2 = null;
				} else if (root1.data > root2.data) {
					s2.pop();
					root2 = root2.right;
					root1 = null;
				}
			} else
				break;
		}
		return result;
	}

	public BinaryTreeNode inorderSuccessor(BinaryTreeNode root, BinaryTreeNode x) {
		if (x.right != null) {
			return minValue(x.right);
		}
		BinaryTreeNode succ = null;
		while (root != null) {
			if (x.data < root.data) {
				succ = root;
				root = root.left;
			} else if (x.data > root.data) {
				root = root.right;
			} else {
				break;
			}
		}
		return succ;
	}

	private BinaryTreeNode minValue(BinaryTreeNode n) {
		BinaryTreeNode node = n;
		while (node.left != null) {
			node = node.left;
		}
		return node;
	}

	public static float findMedian(BinaryTreeNode root) {
		int n = countNodes(root);
		if (n % 2 != 0) {
			int mid[] = new int[1];
			mid[0] = 0;
			find(root, 0, 1 + n / 2, mid);
			return (float) (mid[0]);
		} else {
			int mid1[] = new int[1];
			int mid2[] = new int[1];
			mid1[0] = 0;
			mid2[0] = 0;
			find(root, 0, n / 2, mid1);
			find(root, 0, 1 + n / 2, mid2);
			if ((mid1[0] + mid2[0]) % 2 == 0)
				return (float) (mid1[0] + mid2[0]) / 2;
			else
				return ((float) (mid1[0] + mid2[0]) / 2);
		}
	}

	public static int countNodes(BinaryTreeNode n) {
		if (n == null)
			return 0;
		return 1 + countNodes(n.left) + countNodes(n.right);
	}

	public static int find(BinaryTreeNode n, int serialNo, int target, int value[]) {
		if (n == null)
			return serialNo;
		serialNo = find(n.left, serialNo, target, value);
		serialNo++;
		if (serialNo == target)
			value[0] = n.data;
		serialNo = find(n.right, serialNo, target, value);
		return serialNo;
	}

	public static boolean isBST(BinaryTreeNode root) {
		return isBSTUtil(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
	}

	private static boolean isBSTUtil(BinaryTreeNode node, int min, int max) {
		if (node == null)
			return true;
		if (node.data < min || node.data > max)
			return false;
		return (isBSTUtil(node.left, min, node.data - 1) && isBSTUtil(node.right, node.data + 1, max));
	}

	public BinaryTreeNode modify(BinaryTreeNode root) {
		if (root == null)
			return null;
		generateGreaterTree(root, 0);
		return root;
	}

	private static int generateGreaterTree(BinaryTreeNode root, int sum) {
		if (root.right != null)
			sum = generateGreaterTree(root.right, sum);
		root.data += sum;
		if (root.left != null)
			return generateGreaterTree(root.left, root.data);
		else
			return root.data;
	}

	public static BinaryTreeNode deleteNode(BinaryTreeNode root, int X) {
		if (root == null)
			return root;
		if (root.data > X) {
			root.left = deleteNode(root.left, X);
			return root;
		} else if (root.data < X) {
			root.right = deleteNode(root.right, X);
			return root;
		}
		if (root.left == null) {
			BinaryTreeNode temp = root.right;
			return temp;
		} else if (root.right == null) {
			BinaryTreeNode temp = root.left;
			return temp;
		} else {
			BinaryTreeNode succParent = root;
			BinaryTreeNode succ = root.right;
			while (succ.left != null) {
				succParent = succ;
				succ = succ.left;
			}
			if (succParent != root)
				succParent.left = succ.right;
			else
				succParent.right = succ.right;
			root.data = succ.data;
			return root;
		}
	}

	static int largestBst(Node root) {
		return bst(root).size;
	}

	static Node1 bst(Node root) {
		Node1 x;
		if (root == null) {
			x = new Node1(true, 0, 1000000, 0);
			return x;
		}
		Node1 left = bst(root.left);
		Node1 right = bst(root.right);
		if (left.isBst && right.isBst && root.data > left.maxi && root.data < right.mini)
			x = new Node1(true, 1 + left.size + right.size, Math.min(root.data, left.mini),
					Math.max(root.data, right.maxi));
		else
			x = new Node1(false, Math.max(left.size, right.size), 1000000, 0);
		return x;
	}

	public static int minDiff(Node root, int K) {
		if (root == null)
			return Integer.MAX_VALUE;
		if (root.data == K)
			return 0;
		if (root.data > K)
			return Math.min(Math.abs(root.data - K), minDiff(root.left, K));
		return Math.min(Math.abs(root.data - K), minDiff(root.right, K));
	}

	public List<Integer> merge(BinaryTreeNode root1, BinaryTreeNode root2) {
		List<Integer> one = new ArrayList<Integer>();
		storeInOrder(root1, one);
		List<Integer> two = new ArrayList<Integer>();
		storeInOrder(root2, two);
		List<Integer> result = mergeSort(one, two);
		return result;
	}

	private List<Integer> mergeSort(List<Integer> one, List<Integer> two) {
		int i = 0, j = 0;
		List<Integer> result = new ArrayList<Integer>();
		while (i < one.size() && j < two.size()) {
			if (one.get(i) < two.get(j)) {
				result.add(one.get(i));
				i++;
			} else {
				result.add(two.get(j));
				j++;
			}
		}
		while (i < one.size()) {
			result.add(one.get(i));
			i++;
		}
		while (j < two.size()) {
			result.add(two.get(j));
			j++;
		}
		return result;
	}

	private static void storeInOrder(BinaryTreeNode root, List<Integer> list) {
		if (root != null) {
			storeInOrder(root.left, list);
			list.add(root.data);
			storeInOrder(root.right, list);
		}
	}

	public int KthSmallestElement(BinaryTreeNode root, int K) {
		int count = 0;
		int kSmall = Integer.MIN_VALUE;
		BinaryTreeNode curr = root;
		while (curr != null) {
			if (curr.left == null) {
				count++;
				if (count == K) {
					kSmall = curr.data;
				}
				curr = curr.right;
			} else {
				BinaryTreeNode prev = curr.left;
				while (prev.right != null && prev.right != curr) {
					prev = prev.right;
				}
				if (prev.right == null) {
					prev.right = curr;
					curr = curr.left;
				} else {
					prev.right = null;
					count++;
					if (count == K) {
						kSmall = curr.data;
					}
					curr = curr.right;
				}
			}
		}
		return kSmall;
	}

	public int isPairPresent(Node root, int target) {
		return pairUtil(root, target, new HashSet<Integer>());
	}

	private static int pairUtil(Node root, int target, HashSet<Integer> set) {
		if (root == null) {
			return 0;
		}
		if (pairUtil(root.left, target, set) == 1) {
			return 1;
		}
		if (set.contains(target - root.data)) {
			return 1;
		} else {
			set.add(root.data);
		}
		return pairUtil(root.right, target, set);
	}

	public static int numTrees(int n) {
		long dp[] = new long[n + 1];
		dp[0] = 1;
		dp[1] = 1;
		long mod = 1000000007;
		for (int i = 2; i <= n; i++) {
			dp[i] = 0;
			for (int j = 1; j <= i; j++) {
				dp[i] = (dp[i] + (dp[j - 1] * dp[i - j]) % mod) % mod;
			}
		}
		return (int) dp[n];
	}

	Node first, middle, last, prev;

	public Node correctBST(Node root) {
		first = middle = last = prev = null;
		correctBSTUtil(root);
		if (first != null && last != null) {
			int temp = first.data;
			first.data = last.data;
			last.data = temp;
		}
		else if (first != null && middle != null) {
			int temp = first.data;
			first.data = middle.data;
			middle.data = temp;
		}
		return root;
	}

	void correctBSTUtil(Node root) {
		if (root != null) {
			correctBSTUtil(root.left);
			if (prev != null && root.data < prev.data) {
				if (first == null) {
					first = prev;
					middle = root;
				} else
					last = root;
			}
			prev = root;
			correctBSTUtil(root.right);
		}
	}

}

class BinaryTreeNode {
	int data;
	BinaryTreeNode left, right;

	public BinaryTreeNode(int d) {
		data = d;
		left = right = null;
	}
}

class Node1 {
	boolean isBst;
	int size;
	int mini;
	int maxi;

	public Node1(boolean isBst, int size, int mini, int maxi) {
		this.isBst = isBst;
		this.size = size;
		this.mini = mini;
		this.maxi = maxi;
	}
}
