package com.test.leetcode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;

public class CodecSolution {
	public static void main(String[] args) {
		Codec codec = new Codec();
		TreeNode root = new TreeNode(1);
		root.left = new TreeNode(2);
		root.right = new TreeNode(3);
		root.right.left = new TreeNode(4);
		root.right.right = new TreeNode(5);
		System.out.println(preorder(codec.deserialize(codec.serialize(root))));
	}

	static ArrayList<Integer> preorder(TreeNode root) {
		ArrayList<Integer> preOrder = new ArrayList<Integer>();
		preOrderUtil(root, preOrder);
		return preOrder;
	}

	private static void preOrderUtil(TreeNode root, ArrayList<Integer> preOrder) {
		if (root != null) {
			preOrder.add(root.val);
			preOrderUtil(root.left, preOrder);
			preOrderUtil(root.right, preOrder);
		}
	}
}

class Codec {

	public String serialize(TreeNode root) {
		return serial(new StringBuilder(), root).toString();
	}

	private StringBuilder serial(StringBuilder str, TreeNode root) {
		if (root == null)
			return str.append("#");
		str.append(root.val).append(",");
		serial(str, root.left).append(",");
		serial(str, root.right);
		return str;
	}

	public TreeNode deserialize(String data) {
		return deserial(new LinkedList<>(Arrays.asList(data.split(","))));
	}

	private TreeNode deserial(Queue<String> q) {
		String val = q.poll();
		if ("#".equals(val))
			return null;
		TreeNode root = new TreeNode(Integer.valueOf(val));
		root.left = deserial(q);
		root.right = deserial(q);
		return root;
	}

}
