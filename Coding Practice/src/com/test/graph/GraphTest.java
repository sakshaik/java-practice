package com.test.graph;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class GraphTest {

	public ArrayList<Integer> dfsOfGraph(int V, ArrayList<ArrayList<Integer>> adj) {
		boolean[] vis = new boolean[V];
		ArrayList<Integer> ans = new ArrayList<Integer>();
		dfs(0, vis, ans, adj);
		return ans;
	}

	private void dfs(int ch, boolean[] vis, ArrayList<Integer> ans, ArrayList<ArrayList<Integer>> adj) {
		vis[ch] = true;
		ans.add(ch);
		for (int i = 0; i < adj.get(ch).size(); i++) {
			if (!vis[adj.get(ch).get(i)]) {
				dfs(adj.get(ch).get(i), vis, ans, adj);
			}
		}
	}

	public ArrayList<Integer> bfsOfGraph(int V, ArrayList<ArrayList<Integer>> adj) {
		boolean visited[] = new boolean[V];
		int s = 0;
		visited[s] = true;
		ArrayList<Integer> res = new ArrayList<>();
		LinkedList<Integer> q = new LinkedList<Integer>();
		q.add(s);
		while (q.size() != 0) {
			s = q.poll();
			res.add(s);
			Iterator<Integer> i = adj.get(s).listIterator();
			while (i.hasNext()) {
				int n = i.next();
				if (!visited[n]) {
					visited[n] = true;
					q.add(n);
				}
			}
		}
		return res;
	}

	public int orangesRotting(int[][] grid) {
		int ct = 0, res = -1;
		Queue<ArrayList<Integer>> q = new LinkedList<>();
		int[] dx = { -1, 1, 0, 0 };
		int[] dy = { 0, 0, -1, 1 };
		for (int i = 0; i < grid.length; i++) {
			for (int j = 0; j < grid[0].length; j++) {
				if (grid[i][j] > 0)
					ct++;
				if (grid[i][j] == 2) {
					ArrayList<Integer> temp = new ArrayList<>();
					temp.add(i);
					temp.add(j);
					q.add(temp);
				}
			}
		}

		while (!q.isEmpty()) {
			res++;
			int size = q.size();
			for (int k = 0; k < size; k++) {
				ArrayList<Integer> cur = q.poll();
				ct--;
				for (int i = 0; i < 4; i++) {
					int x = cur.get(0) + dx[i], y = cur.get(1) + dy[i];
					if (x >= grid.length || x < 0 || y >= grid[0].length || y < 0 || grid[x][y] != 1)
						continue;
					grid[x][y] = 2;
					ArrayList<Integer> temp = new ArrayList<>();
					temp.add(x);
					temp.add(y);
					q.add(temp);
				}
			}
		}
		if (ct == 0)
			return Math.max(0, res);
		return -1;
	}

	// Function to detect cycle in a directed graph.
	public boolean isCyclic(int V, ArrayList<ArrayList<Integer>> adj) {
		// marking all vertices as not visited and not a part of recursion stack
		boolean[] visited = new boolean[V];
		boolean[] recStack = new boolean[V];
		// calling the recursive helper function to detect cycle in
		// different DFS trees.
		for (int i = 0; i < V; i++)
			if (isCyclicUtil(i, visited, recStack, adj))
				return true;
		return false;
	}

	boolean isCyclicUtil(int i, boolean[] visited, boolean[] recStack, ArrayList<ArrayList<Integer>> adj) {
		// marking the current node as visited and part of recursion stack.
		if (recStack[i])
			return true;
		if (visited[i])
			return false;
		visited[i] = true;
		recStack[i] = true;
		List<Integer> children = adj.get(i);
		// calling function recursively for all the vertices
		// adjacent to this vertex.
		for (Integer c : children)
			if (isCyclicUtil(c, visited, recStack, adj))
				return true;
		// removing the vertex from recursion stack.
		recStack[i] = false;
		return false;
	}

	// Function to detect cycle in an undirected graph.
	public boolean isCycle(int V, ArrayList<ArrayList<Integer>> adj) {
		// using a boolean list to mark all the vertices as not visited.
		Boolean visited[] = new Boolean[V];
		for (int i = 0; i < V; i++)
			visited[i] = false;
		// iterating over all the vertices.
		for (int u = 0; u < V; u++) {
			// if vertex is not visited, we call the function to detect cycle.
			if (!visited[u])
				// if cycle is found, we return true.
				if (isCyclicUtil(u, visited, -1, adj))
					return true;
		}
		return false;
	}

	Boolean isCyclicUtil(int v, Boolean visited[], int parent, ArrayList<ArrayList<Integer>> adj) {
		// marking the current vertex as visited.
		visited[v] = true;
		Integer i;
		Iterator<Integer> it = adj.get(v).iterator();
		while (it.hasNext()) {
			i = it.next();
			// if an adjacent is not visited, then calling function
			// recursively for that adjacent vertex.
			if (!visited[i]) {
				if (isCyclicUtil(i, visited, v, adj))
					return true;
			}
			// if an adjacent is visited and it is not a parent of current
			// vertex then there is a cycle and we return true.
			else if (i != parent)
				return true;
		}
		return false;
	}

	public int minSwaps(int nums[]) {
		int len = nums.length;
		HashMap<Integer, Integer> map = new HashMap<>();
		for (int i = 0; i < len; i++)
			map.put(nums[i], i);
		Arrays.sort(nums);
		// To keep track of visited elements. Initialize
		// all elements as not visited or false.
		boolean[] visited = new boolean[len];
		Arrays.fill(visited, false);
		// Initialize result
		int ans = 0;
		for (int i = 0; i < len; i++) {
			// already swapped and corrected or
			// already present at correct pos
			if (visited[i] || map.get(nums[i]) == i)
				continue;
			int j = i, cycle_size = 0;
			while (!visited[j]) {
				visited[j] = true;
				// move to next node
				j = map.get(nums[j]);
				cycle_size++;
			}
			// Update answer by adding current cycle.
			if (cycle_size > 0) {
				ans += (cycle_size - 1);
			}
		}
		return ans;
	}

	// Function to find a Mother Vertex in the Graph.
	public int findMotherVertex(int V, ArrayList<ArrayList<Integer>> adj) {
		// boolean list to mark the visited nodes and initially all are
		// initialized as not visited.
		boolean[] visited = new boolean[V];
		// variable to store last finished vertex (or mother vertex).
		int v = -1;
		// iterating over all the vertices
		for (int i = 0; i < V; i++) {
			// if current vertex is not visited, we call the dfs
			// function and then update the variable v.
			if (!visited[i]) {
				DFSUtil(adj, i, visited);
				v = i;
			}
		}
		// we reset all the vertices as not visited.
		boolean[] check = new boolean[V];
		// calling the dfs function to do DFS beginning from v to check
		// if all vertices are reachable from it or not.
		DFSUtil(adj, v, check);
		// iterating on boolean list and returning -1 if
		// any vertex is not visited.
		for (boolean val : check) {
			if (!val) {
				return -1;
			}
		}
		// returning mother vertex.
		return v;
	}

	static void DFSUtil(ArrayList<ArrayList<Integer>> g, int v, boolean[] visited) {
		// marking current vertex as visited.
		visited[v] = true;
		// iterating over the adjacent vertices.
		for (int x : g.get(v)) {
			// if any vertex is not visited, we call dfs function recursively.
			if (!visited[x]) {
				DFSUtil(g, x, visited);
			}
		}
	}

	/*
	 * Given a grid of size n*n filled with 0, 1, 2, 3. Check whether there is a
	 * path possible from the source to destination. You can traverse up, down,
	 * right and left. The description of cells is as follows:
	 * 
	 * A value of cell 1 means Source. A value of cell 2 means Destination. A value
	 * of cell 3 means Blank cell. A value of cell 0 means Wall. Note: There are
	 * only a single source and a single destination.
	 */

	// Function to find whether a path exists from the source to destination.
	public boolean is_Possible(int[][] grid) {
		int n = grid.length;
		int m = grid[0].length;
		// using boolean array to mark visited cells.
		boolean vis[][] = new boolean[n][m];
		int sx = -1, sy = -1, dx = -1, dy = -1;
		// traversing all the cells of the matrix.
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				// storing the source and destination indexes.
				if (grid[i][j] == 1) {
					sx = i;
					sy = j;
				}
				if (grid[i][j] == 2) {
					dx = i;
					dy = j;
				}
			}
		}
		// calling function to check if path exists and returning the result.
		return dfs(sx, sy, dx, dy, vis, grid);
	}

	private boolean dfs(int sx, int sy, int dx, int dy, boolean[][] vis, int[][] grid) {
		// if source and destination indexes are same, we return true.
		if (sx == dx && sy == dy)
			return true;
		// marking the cell as visited.
		vis[sx][sy] = true;
		boolean ans = false;
		// calling function recursively for adjacent cells.
		if (sx - 1 >= 0 && grid[sx - 1][sy] != 0 && !vis[sx - 1][sy])
			ans |= dfs(sx - 1, sy, dx, dy, vis, grid);
		if (sx + 1 < grid.length && grid[sx + 1][sy] != 0 && !vis[sx + 1][sy])
			ans |= dfs(sx + 1, sy, dx, dy, vis, grid);
		if (sy - 1 >= 0 && grid[sx][sy - 1] != 0 && !vis[sx][sy - 1])
			ans |= dfs(sx, sy - 1, dx, dy, vis, grid);
		if (sy + 1 < grid[0].length && grid[sx][sy + 1] != 0 && !vis[sx][sy + 1])
			ans |= dfs(sx, sy + 1, dx, dy, vis, grid);
		return ans;
	}

}
