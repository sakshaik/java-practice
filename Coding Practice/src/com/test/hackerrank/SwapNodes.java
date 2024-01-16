package com.test.hackerrank;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.stream.Collectors;

public class SwapNodes {

	public static void main(String[] args) {
		List<List<Integer>> indexes = Arrays.asList(Arrays.asList(2, 3), Arrays.asList(-1, -1), Arrays.asList(-1, -1));
		List<Integer> queries = Arrays.asList(1, 1);
		SwapNodes.swapNodes(indexes, queries).stream().collect(Collectors.toList()).forEach(x -> System.out.println(x));
	}

	static List<List<Integer>> swapNodes(List<List<Integer>> indexes, List<Integer> queries) {
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		Node root = new Node(1);
		Queue<Node> q = new LinkedList<>();
		q.add(root);
		for (int i = 0; i < indexes.size() && !q.isEmpty(); i++) {
			Node temp = q.remove();
			if (indexes.get(i).get(0) != -1) {
				Node left = new Node(indexes.get(i).get(0));
				temp.left = left;
				q.add(left);
			} else {
				temp.left = null;
			}

			if (indexes.get(i).get(1) != -1) {
				Node right = new Node(indexes.get(i).get(1));
				temp.right = right;
				q.add(right);
			} else {
				temp.right = null;
			}
		}

		for (int i = 0; i < queries.size(); i++) {
			swap(root, queries.get(i), 1);
			List<Integer> temp = new ArrayList<>();
			inorder(root, temp);
			result.add(temp);
		}
		return result;
	}

	static void swap(Node node, int k, int level) {
		if (node == null)
			return;
		if (level % k == 0) {
			Node temp = node.left;
			node.left = node.right;
			node.right = temp;
		}
		swap(node.left, k, level + 1);
		swap(node.right, k, level + 1);
	}

	static void inorder(Node node, List<Integer> lst) {
		if (node != null) {
			inorder(node.left, lst);
			lst.add(node.data);
			inorder(node.right, lst);
		}
	}

}

class Node {
	Node left;
	Node right;
	int data;

	Node(int nodeData) {
		this.data = nodeData;
		left = right = null;
	}
}