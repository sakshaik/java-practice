package com.test.leetcode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class AccountMergeSolution {

	/*
	 * Given a list of accounts where each element accounts[i] is a list of strings,
	 * where the first element accounts[i][0] is a name, and the rest of the
	 * elements are emails representing emails of the account.
	 * 
	 * Now, we would like to merge these accounts. Two accounts definitely belong to
	 * the same person if there is some common email to both accounts. Note that
	 * even if two accounts have the same name, they may belong to different people
	 * as people could have the same name. A person can have any number of accounts
	 * initially, but all of their accounts definitely have the same name.
	 * 
	 * After merging the accounts, return the accounts in the following format: the
	 * first element of each account is the name, and the rest of the elements are
	 * emails in sorted order. The accounts themselves can be returned in any order.
	 */

	public static void main(String[] args) {
		AccountMergeSolution solution = new AccountMergeSolution();
		List<List<String>> accounts = Arrays.asList(
				Arrays.asList("John", "johnsmith@mail.com", "john_newyork@mail.com"),
				Arrays.asList("John", "johnsmith@mail.com", "john00@mail.com"), Arrays.asList("Mary", "mary@mail.com"),
				Arrays.asList("John", "johnnybravo@mail.com"));
		System.out.println(solution.accountsMerge(accounts));
	}

	public class Pair {
		int idx;
		String str;

		Pair(int idx, String str) {
			this.idx = idx;
			this.str = str;
		}
	}

	static int[] par;

	public List<List<String>> accountsMerge(List<List<String>> accounts) {
		HashMap<String, Pair> map = new HashMap<>();
		int count = 0;
		for (List<String> ls : accounts) {
			for (int i = 1; i < ls.size(); i++) {
				if (map.containsKey(ls.get(i)) == false)
					map.put(ls.get(i), new Pair(count++, ls.get(0)));
			}
		}
		par = new int[count];
		for (int i = 0; i < par.length; i++)
			par[i] = i;
		for (List<String> ls : accounts) {
			for (int i = 2; i < ls.size(); i++) {
				union(map.get(ls.get(1)).idx, map.get(ls.get(i)).idx);
			}
		}
		HashMap<Integer, List<String>> vals = new HashMap<>();
		List<List<String>> ans = new ArrayList<>();

		for (String email : map.keySet()) {
			int pid = map.get(email).idx;
			pid = find(pid);
			if (!vals.containsKey(pid))
				vals.put(pid, new ArrayList<>());
			vals.get(pid).add(email);
		}

		for (int val : vals.keySet()) {
			Collections.sort(vals.get(val));
			List<String> toadd = new ArrayList<>(vals.get(val));
			toadd.add(0, map.get(vals.get(val).get(0)).str);
			ans.add(new ArrayList<>(toadd));
		}

		return ans;
	}

	public static void union(int x, int y) {
		int lx = find(x);
		int ly = find(y);
		if (lx != ly)
			par[lx] = ly;
	}

	public static int find(int x) {
		if (par[x] == x)
			return x;
		int temp = find(par[x]);
		return par[x] = temp;
	}

}
