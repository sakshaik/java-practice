package com.test.practice.geekforgeeks;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

public class HeightOrderSolution {

	public static void main(String[] args) {
		ArrayList<Integer> heights = new ArrayList<Integer>();
		int[] height = { 9, 1, 2, 3, 3, 6, 5 };
		for (int i = 0; i < height.length; i++) {
			heights.add(height[i]);
		}

		ArrayList<Integer> infronts = new ArrayList<Integer>();
		int[] infront = { 0, 0, 0, 0, 1, 1, 2 };
		for (int i = 0; i < infront.length; i++) {
			infronts.add(infront[i]);
		}
		HeightOrderSolution solution = new HeightOrderSolution();
		System.out.println(solution.arrange(heights, infronts, heights.size()));

		ArrayList<Integer> heights2 = new ArrayList<Integer>();
		int[] height2 = { 3, 2, 1 };
		for (int i = 0; i < height2.length; i++) {
			heights2.add(height2[i]);
		}

		ArrayList<Integer> infronts2 = new ArrayList<Integer>();
		int[] infront2 = { 0, 1, 1 };
		for (int i = 0; i < infront2.length; i++) {
			infronts2.add(infront2[i]);
		}
		HeightOrderSolution solution2 = new HeightOrderSolution();
		System.out.println(solution2.arrange(heights2, infronts2, heights2.size()));
	}

	public ArrayList<Integer> arrange(ArrayList<Integer> A, ArrayList<Integer> B, int n) {
		ArrayList<Person> persons = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			persons.add(new Person(A.get(i), B.get(i)));
		}

		Collections.sort(persons, new Comparator<Person>() {
			public int compare(Person a, Person b) {
				if ((a.height < b.height) || (a.height == b.height && a.infront > b.infront))
					return -1;
				return 1;
			}
		});

		CustomTree root = new CustomTree(persons.get(n - 1).height);

		for (int i = n - 2; i >= 0; i--) {
			populateTreeWithPerson(root, persons.get(i));
		}

		ArrayList<Integer> ans = new ArrayList<>();

		inOrderCustomTreeHeights(root, ans);

		return ans;
	}

	private void populateTreeWithPerson(CustomTree node, Person person) {
		if (person.infront < node.leftCount + 1) {
			node.leftCount++;
			if (node.left != null)
				populateTreeWithPerson(node.left, person);
			else
				node.left = new CustomTree(person.height);
		} else {
			person.infront -= node.leftCount + 1;
			if (node.right != null)
				populateTreeWithPerson(node.right, person);
			else
				node.right = new CustomTree(person.height);
		}
	}

	private void inOrderCustomTreeHeights(CustomTree node, ArrayList<Integer> ans) {
		if (node == null)
			return;
		inOrderCustomTreeHeights(node.left, ans);
		ans.add(node.height);
		inOrderCustomTreeHeights(node.right, ans);
	}

	class Person {
		int height, infront;

		Person(int height, int infront) {
			this.height = height;
			this.infront = infront;
		}
	}

	class CustomTree {
		int height, leftCount;
		CustomTree left, right;

		CustomTree(int height) {
			this.height = height;
			left = null;
			right = null;
			leftCount = 0;
		}

	}
}
