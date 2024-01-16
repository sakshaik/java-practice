package com.test.practice.geekforgeeks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Stack;
import java.util.TreeSet;

public class RaceRankingSolution {
	Stack<Integer> ans;
	public void printOrder(ArrayList<Pair> info, int n, int p) {
		TreeSet<String> trs = new TreeSet<String>(); // Taking Set of names
		HashMap<String, Integer> map = new HashMap<String, Integer>(); // map returns index corresponding to name
		ArrayList<ArrayList<Integer>> adj = new ArrayList<ArrayList<Integer>>(); // Creating the graph adjaency list

		for (int i = 0; i < p; i++) {
			// pushing the names into the set
			trs.add(info.get(i).first);
			trs.add(info.get(i).second);
		}

		String[] runners = new String[n]; // returns name corresponding to given index

		int ind = 0;
		for (String s : trs) {
			// Creating the map for the name.
			runners[ind] = s;
			map.put(s, ind);
			// assigning memory to adjaency list
			adj.add(new ArrayList<Integer>());
			ind++;
		}
		int[] independent = new int[n];

		for (int i = 0; i < p; i++) {
			adj.get(map.get(info.get(i).first)).add(map.get(info.get(i).second));
			// how many ansestor of the particular node.
			independent[map.get(info.get(i).second)]++;
		}
		// creating visited array which mark whether the particular node is visited or
		// not.
		boolean[] vis = new boolean[n];
		Arrays.fill(vis, false); // Set initially all the value of vis as false.

		ans = new Stack<Integer>(); // creating Stack

		for (int i = 0; i < n; i++) {
			if (vis[i] == false) {
				// mark i node as visited and call the dfs to assosiated graph with this node.
				vis[i] = true;
				dfs(i, adj, vis);
			}
		}

		// System.out.println(ans);
		while (!ans.isEmpty()) {
			// print the stored order
			System.out.print(runners[ans.peek()] + " ");
			ans.pop();
		}
		System.out.println();
	}

	public void dfs(int nd, ArrayList<ArrayList<Integer>> adj, boolean[] vis) {
		// Traverse all the node of the particular value.
		for (int u : adj.get(nd)) {
			if (vis[u]) {
				continue;
			}
			vis[u] = true;
			dfs(u, adj, vis);
		}
		// push the order in the stack
		ans.push(nd);
	}

}

class Pair {
	String first, second;

	Pair() {
		first = second = null;
	}

	Pair(String f, String s) {
		first = f;
		second = s;
	}
}