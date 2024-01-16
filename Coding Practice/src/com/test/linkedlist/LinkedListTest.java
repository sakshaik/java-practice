package com.test.linkedlist;

import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.TreeSet;

public class LinkedListTest {

	public static void main(String[] args) {
		int[] a = { 1, 2, 3 };
		int[] b = { 1, 2, 8, 6, 2 };

		Node head = new Node(a[0]);
		Node tail = head;
		for (int i = 1; i < a.length; i++) {
			tail.next = new Node(a[i]);
			tail = tail.next;
		}

		Node head2 = new Node(b[0]);
		Node tail2 = head2;
		for (int i = 1; i < b.length; i++) {
			tail2.next = new Node(b[i]);
			tail2 = tail2.next;
		}

		Node head10 = reorderlist(head);
		printList(head10);

		Node head9 = deleteAllOccurances(head, 2);
		printList(head9);

		Node head8 = reverse(head, 2);
		printList(head8);

		Node head7 = zigZagList(head);
		printList(head7);

		Node head6 = removeDuplicates(head);
		printList(head6);

		Node head5 = findUnion(head, head2);
		printList(head5);

		printList(head);
		System.out.println(isPalindrome(head));
		printList(head);

		Node head4 = findIntersection(head, head2);
		printList(head4);

		printList(head);
		head = mergeSort(head);
		printList(head);

		Node head3 = sortedMerge(head, head2);
		printList(head3);
		System.out.println(detectLoop(head));
		printList(head);
		System.out.println(getNthFromLast(head, 2));
		printList(head);
		rearrangeEvenOdd(head);
		System.out.println();
		printList(head);
		head = addOne(head);
		System.out.println();
		printList(head);
		head = segregate(head);
		System.out.println();
		printList(head);
		head = sortedInsert(head, 16);
		System.out.println();
		printList(head);
		head = reverseList(head);
		System.out.println();
		printList(head);
		head = rotate(head, 2);
		System.out.println();
		printList(head);
		head = pairwiseSwap(head);
		System.out.println();
		printList(head);
	}

	public static int getNthFromLast(Node head, int n) {
		int count = 1;
		int result = head.data;
		int size = totalSize(head);
		Node temp = head;
		if (size < n) {
			return -1;
		}
		while (count <= size - n) {
			temp = temp.next;
			result = temp.data;
			count++;
		}
		return result;
	}

	static int countNodes(Node n) {
		int res = 1;
		Node temp = n;
		while (temp.next != n) {
			res++;
			temp = temp.next;
		}
		return res;
	}

	private static int totalSize(Node head) {
		Node temp = head;
		int size = 0;
		while (temp != null) {
			temp = temp.next;
			size++;
		}
		return size;
	}

	public static boolean detectLoop(Node head) {
		Node slow_p = head, fast_p = head;
		int flag = 0;
		while (slow_p != null && fast_p != null && fast_p.next != null) {
			slow_p = slow_p.next;
			fast_p = fast_p.next.next;
			if (slow_p == fast_p) {
				flag = 1;
				break;
			}
		}
		return flag == 1 ? true : false;
	}

	public static int countNodesinLoop(Node head) {
		Node slow_p = head, fast_p = head;
		int result = 0;
		while (slow_p != null && fast_p != null && fast_p.next != null) {
			slow_p = slow_p.next;
			fast_p = fast_p.next.next;
			if (slow_p == fast_p) {
				result = countNodes(slow_p);
			}
		}
		return result;
	}

	private static Node sortedMerge(Node head1, Node head2) {
		Node dummyNode = new Node(0);
		Node tail = dummyNode;
		while (true) {
			if (head1 == null) {
				tail.next = head2;
				break;
			}
			if (head2 == null) {
				tail.next = head1;
				break;
			}
			if (head1.data <= head2.data) {
				tail.next = head1;
				head1 = head1.next;
			} else {
				tail.next = head2;
				head2 = head2.next;
			}
			tail = tail.next;
		}
		return dummyNode.next;
	}

	static Node mergeSort(Node head) {
		if (head.next == null)
			return head;
		Node mid = findMid(head);
		Node head2 = mid.next;
		mid.next = null;
		Node newHead1 = mergeSort(head);
		Node newHead2 = mergeSort(head2);
		Node finalHead = merge(newHead1, newHead2);
		return finalHead;
	}

	Node mergeKList(Node[] arr, int K) {
		Node head = null, last = null;
		PriorityQueue<Node> pq = new PriorityQueue<>(new Comparator<Node>() {
			public int compare(Node a, Node b) {
				return a.data - b.data;
			}
		});
		for (int i = 0; i < K; i++)
			if (arr[i] != null)
				pq.add(arr[i]);
		while (!pq.isEmpty()) {
			Node top = pq.peek();
			pq.remove();
			if (top.next != null)
				pq.add(top.next);
			if (head == null) {
				head = top;
				last = top;
			} else {
				last.next = top;
				last = top;
			}
		}
		return head;
	}

	public static Node findIntersection(Node head1, Node head2) {
		if (head1 == null || head2 == null)
			return null;

		if (head1.data < head2.data)
			return findIntersection(head1.next, head2);

		if (head1.data > head2.data)
			return findIntersection(head1, head2.next);

		Node temp = new Node(head1.data);
		temp.next = findIntersection(head1.next, head2.next);
		return temp;
	}

	int intersectPoint(Node head1, Node head2) {
		Node ptr1 = head1;
		Node ptr2 = head2;
		if (ptr1 == null || ptr2 == null) {
			return -1;
		}
		while (ptr1 != ptr2) {
			ptr1 = ptr1.next;
			ptr2 = ptr2.next;
			if (ptr1 == ptr2) {
				return ptr1.data;
			}
			if (ptr1 == null) {
				ptr1 = head2;
			}
			if (ptr2 == null) {
				ptr2 = head1;
			}
		}
		return ptr1.data;
	}

	private static Node findMid(Node head) {
		Node slow = head, fast = head.next;
		while (fast != null && fast.next != null) {
			slow = slow.next;
			fast = fast.next.next;
		}
		return slow;
	}

	static Node merge(Node head1, Node head2) {
		Node merged = new Node(-1);
		Node temp = merged;
		while (head1 != null && head2 != null) {
			if (head1.data < head2.data) {
				temp.next = head1;
				head1 = head1.next;
			} else {
				temp.next = head2;
				head2 = head2.next;
			}
			temp = temp.next;
		}
		while (head1 != null) {
			temp.next = head1;
			head1 = head1.next;
			temp = temp.next;
		}
		while (head2 != null) {
			temp.next = head2;
			head2 = head2.next;
			temp = temp.next;
		}
		return merged.next;
	}

	public static void removeLoop(Node head) {
		if (head == null || head.next == null)
			return;

		Node slow = head, fast = head;
		slow = slow.next;
		fast = fast.next.next;

		while (fast != null && fast.next != null) {
			if (slow == fast)
				break;

			slow = slow.next;
			fast = fast.next.next;
		}

		if (slow == fast) {
			slow = head;
			if (slow != fast) {
				while (slow.next != fast.next) {
					slow = slow.next;
					fast = fast.next;
				}
				fast.next = null;
			} else {
				while (fast.next != slow) {
					fast = fast.next;
				}
				fast.next = null;
			}
		}
	}

	private static void rearrangeEvenOdd(Node head) {
		if (head == null)
			return;

		Node odd = head;
		Node even = head.next;

		Node evenFirst = even;

		while (1 == 1) {
			if (odd == null || even == null || (even.next) == null) {
				odd.next = evenFirst;
				break;
			}
			odd.next = even.next;
			odd = even.next;
			if (odd.next == null) {
				even.next = null;
				odd.next = evenFirst;
				break;
			}
			even.next = odd.next;
			even = odd.next;
		}
	}

	public static Node addOne(Node head) {
		Node ln = head;

		if (head.next == null) {
			head.data += 1;
			return head;
		}

		Node t = head;

		while (t.next != null) {
			if (t.data != 9) {
				ln = t;
			}
			t = t.next;
		}
		if (t.data == 9 && ln != null) {
			t = ln;
			t.data += 1;
			t = t.next;
			while (t != null) {
				t.data = 0;
				t = t.next;
			}
		} else {
			t.data += 1;
		}
		return head;
	}

	static Node segregate(Node head) {
		if (head == null || head.next == null) {
			return head;
		}
		Node zeroD = new Node(0);
		Node oneD = new Node(0);
		Node twoD = new Node(0);

		Node zero = zeroD, one = oneD, two = twoD;
		Node curr = head;
		while (curr != null) {
			if (curr.data == 0) {
				zero.next = curr;
				zero = zero.next;
				curr = curr.next;
			} else if (curr.data == 1) {
				one.next = curr;
				one = one.next;
				curr = curr.next;
			} else {
				two.next = curr;
				two = two.next;
				curr = curr.next;
			}
		}
		zero.next = (oneD.next != null) ? (oneD.next) : (twoD.next);
		one.next = twoD.next;
		two.next = null;
		head = zeroD.next;
		return head;
	}

	public static Node sortedInsert(Node head, int key) {
		Node newData = new Node(key);
		if (head == null || head.data >= newData.data) {
			newData.next = head;
			head = newData;
		} else {
			Node temp = head;
			while (temp.next != null && temp.next.data < newData.data) {
				temp = temp.next;
			}
			newData.next = temp.next;
			temp.next = newData;
		}
		return head;
	}

	public static Node sortedInsertCircular(Node head, int data) {
		Node new_node = new Node(data);
		Node current = head;
		if (current == null) {
			new_node.next = new_node;
			head = new_node;
		} else if (current.data >= new_node.data) {
			while (current.next != head)
				current = current.next;
			current.next = new_node;
			new_node.next = head;
			head = new_node;
		} else {
			while (current.next != head && current.next.data < new_node.data)
				current = current.next;
			new_node.next = current.next;
			current.next = new_node;
		}
		return head;
	}

	public static Node reverseList(Node head) {
		Node prev = null;
		Node current = head;
		Node next = null;
		while (current != null) {
			next = current.next;
			current.next = prev;
			prev = current;
			current = next;
		}
		head = prev;
		return head;
	}

	public static Node rotate(Node head, int k) {
		if (k == 0)
			return head;
		Node current = head;
		int count = 1;
		while (count < k && current != null) {
			current = current.next;
			count++;
		}
		if (current == null)
			return head;
		Node kthNode = current;
		while (current.next != null)
			current = current.next;
		current.next = head;
		head = kthNode.next;
		kthNode.next = null;
		return head;
	}

	public static Node pairwiseSwap(Node head) {
		if (head == null || head.next == null) {
			return head;
		}

		Node prev = head;
		Node curr = head.next;

		head = curr;
		while (true) {
			Node next = curr.next;
			curr.next = prev;
			if (next == null || next.next == null) {
				prev.next = next;
				break;
			}
			prev.next = next.next;
			prev = next;
			curr = prev.next;
		}
		return head;
	}

	public static void printList(Node node) {
		while (node != null) {
			System.out.print(node.data + "->");
			node = node.next;
		}
		System.out.println();
	}

	public static boolean isPalindrome(Node head) {
		Node slow_ptr = head;
		Node fast_ptr = head;
		Node prev_of_slow_ptr = head;
		Node midnode = null;
		Node second_half = null;
		boolean res = true;

		if (head != null && head.next != null) {
			while (fast_ptr != null && fast_ptr.next != null) {
				fast_ptr = fast_ptr.next.next;
				prev_of_slow_ptr = slow_ptr;
				slow_ptr = slow_ptr.next;
			}
			if (fast_ptr != null) {
				midnode = slow_ptr;
				slow_ptr = slow_ptr.next;
			}
			second_half = slow_ptr;
			prev_of_slow_ptr.next = null;
			second_half = reverseList(second_half);
			res = compareLists(head, second_half);

			second_half = reverseList(second_half);

			if (midnode != null) {
				prev_of_slow_ptr.next = midnode;
				midnode.next = second_half;
			} else
				prev_of_slow_ptr.next = second_half;
		}
		return res;
	}

	private static boolean compareLists(Node head1, Node head2) {
		Node temp1 = head1;
		Node temp2 = head2;
		while (temp1 != null && temp2 != null) {
			if (temp1.data == temp2.data) {
				temp1 = temp1.next;
				temp2 = temp2.next;
			} else
				return false;
		}
		if (temp1 == null && temp2 == null)
			return true;
		return false;
	}

	public static Node findUnion(Node head1, Node head2) {
		Node cur = null, start = null;
		TreeSet<Integer> s = new TreeSet<Integer>();
		while (head1 != null) {
			s.add(head1.data);
			head1 = head1.next;
		}
		while (head2 != null) {
			s.add(head2.data);
			head2 = head2.next;
		}
		for (Integer i : s) {
			Node ptr = new Node(i);
			if (start == null) {
				start = ptr;
				cur = ptr;
			} else {
				cur.next = ptr;
				cur = ptr;
			}
		}
		return start;
	}

	public static Node removeDuplicates(Node head) {
		HashSet<Integer> hs = new HashSet<>();
		Node current = head;
		Node prev = null;
		while (current != null) {
			int curval = current.data;
			if (hs.contains(curval)) {
				prev.next = current.next;
			} else {
				hs.add(curval);
				prev = current;
			}
			current = current.next;
		}
		return head;
	}

	static Node zigZagList(Node head) {
		boolean flag = true;
		int temp = 0;
		Node current = head;
		while (current != null && current.next != null) {
			if (flag == true) {
				if (current.data > current.next.data) {
					temp = current.data;
					current.data = current.next.data;
					current.next.data = temp;
				}
			} else {
				if (current.data < current.next.data) {
					temp = current.data;
					current.data = current.next.data;
					current.next.data = temp;
				}
			}
			current = current.next;
			flag = !(flag);
		}
		return head;
	}

	static int countPairs(LinkedList<Integer> head1, LinkedList<Integer> head2, int x) {
		int count = 0;
		HashSet<Integer> us = new HashSet<Integer>();
		Iterator<Integer> itr1 = head1.iterator();
		while (itr1.hasNext()) {
			us.add(itr1.next());
		}
		Iterator<Integer> itr2 = head2.iterator();
		while (itr2.hasNext()) {
			if (!(us.add(x - itr2.next())))
				count++;
		}
		return count;
	}

	public static Node reverse(Node node, int k) {
		Node prev = null;
		Node curr = node;
		Node temp = null;
		Node tail = null;
		Node newHead = null;
		Node join = null;
		int t = 0;

		while (curr != null) {
			t = k;
			join = curr;
			prev = null;
			while (curr != null && t-- != 0) {
				temp = curr.next;
				curr.next = prev;
				prev = curr;
				curr = temp;
			}

			if ((newHead == null))
				newHead = prev;
			if (tail != null)
				tail.next = prev;
			tail = join;
		}
		return newHead;
	}

	public static Node copyList(Node head) {
		Node curr = head, temp = null;
		while (curr != null) {
			temp = curr.next;
			curr.next = new Node(curr.data);
			curr.next.next = temp;
			curr = temp;
		}
		curr = head;

		while (curr != null) {
			if (curr.next != null)
				curr.next.arb = (curr.arb != null) ? curr.arb.next : curr.arb;
			curr = curr.next.next;
		}

		Node original = head, copy = head.next;
		temp = copy;
		while (original != null) {
			original.next = original.next.next;
			copy.next = (copy.next != null) ? copy.next.next : copy.next;
			original = original.next;
			copy = copy.next;
		}
		return temp;
	}

	void deleteNode(Node del) {
		Node temp = del.next;
		del.data = temp.data;
		del.next = temp.next;
		temp = null;
	}

	public static Node addPolynomial(Node p1, Node p2) {

		Node a = p1, b = p2, newHead = new Node(0, 0), c = newHead;

		while (a != null || b != null) {
			if (a == null) {
				c.next = b;
				break;
			} else if (b == null) {
				c.next = a;
				break;
			} else if (a.pow == b.pow) {
				c.next = new Node(a.coeff + b.coeff, a.pow);
				a = a.next;
				b = b.next;
			} else if (a.pow > b.pow) {
				c.next = new Node(a.coeff, a.pow);
				a = a.next;
			} else if (a.pow < b.pow) {
				c.next = new Node(b.coeff, b.pow);
				b = b.next;
			}
			c = c.next;
		}
		return newHead.next;
	}

	public static Node compute(Node head) {
		if (head == null || head.next == null)
			return head;
		Node nextNode = compute(head.next);

		if (nextNode.data > head.data)
			return nextNode;
		head.next = nextNode;

		return head;
	}

	public static Node flatten(Node root) {
		if (root == null || root.next == null)
			return root;
		root.next = flatten(root.next);
		root = merge(root, root.next);
		return root;
	}

	public static Node swapkthnode(Node head, int num, int K) {
		if (num < K || 2 * K - 1 == num)
			return head;

		Node x = head;
		Node x_prev = null;
		for (int i = 1; i < K; i++) {
			x_prev = x;
			x = x.next;
		}

		Node y = head;
		Node y_prev = null;
		for (int i = 1; i < num - K + 1; i++) {
			y_prev = y;
			y = y.next;
		}

		if (x_prev != null)
			x_prev.next = y;

		if (y_prev != null)
			y_prev.next = x;

		Node temp = x.next;
		x.next = y.next;
		y.next = temp;

		if (K == 1)
			head = y;

		if (K == num)
			head = x;

		return head;
	}

	public static void rearrange(Node odd) {
		if (odd == null || odd.next == null || odd.next.next == null) {
			return;
		}
		Node even = odd.next;
		odd.next = odd.next.next;
		odd = odd.next;
		even.next = null;
		while (odd.next != null) {
			Node temp = odd.next.next;
			odd.next.next = even;
			even = odd.next;
			odd.next = temp;
			if (temp != null) {
				odd = temp;
			}
		}
		odd.next = even;
	}

	public static Node deleteAllOccurances(Node head, int x) {
		Node temp = head, prev = null;
		while (temp != null && temp.data == x) {
			head = temp.next;
			temp = head;
		}
		while (temp != null) {
			while (temp != null && temp.data != x) {
				prev = temp;
				temp = temp.next;
			}
			if (temp == null)
				return head;
			prev.next = temp.next;
			temp = prev.next;
		}
		return head;
	}

	Node deleteNode(Node head, int x) {
		if (head == null || x <= 0) {
			return head;
		}

		Node temp = head;
		for (int i = 1; i < x && temp != null; i++) {
			temp = temp.next;
		}

		if (temp == null || head == null) {
			return null;
		}

		if (head == temp)
			head = temp.next;

		if (temp.next != null)
			temp.next.prev = temp.prev;

		if (temp.prev != null)
			temp.prev.next = temp.next;

		temp = null;
		return head;
	}

	public static Node reorderlist(Node head) {
		int n = 0;
		Node cur = head;
		while (cur != null) {
			n++;
			cur = cur.next;
		}

		Node head1 = head;
		Node head2 = head;
		Node prev = null;
		int w = n / 2;
		if (n % 2 == 1) {
			w++;
		}

		for (int i = 0; i < w; i++) {
			prev = head2;
			head2 = head2.next;
		}

		if (prev != null) {
			prev.next = null;
		}

		head2 = reverseList(head2);

		cur = head1;

		for (int i = 0; i < n / 2; i++) {
			Node temp = cur.next;
			cur.next = head2;
			Node temp2 = head2.next;
			head2.next = temp;
			cur = temp;
			head2 = temp2;
		}
		return head1;
	}

	public Node removeNthFromEnd(Node head, int n) {
		Node fast = head;
		for (int i = 1; i < n; i++)
			fast = fast.next;
		Node slow = head;
		Node prev = null;
		while (fast.next != null) {
			prev = slow;
			slow = slow.next;
			fast = fast.next;
		}
		if (prev == null)
			head = head.next;
		else
			prev.next = slow.next;
		return head;
	}

}

class Node {
	int data;
	int pow;
	int coeff;
	Node next, arb, prev;

	Node(int x) {
		data = x;
		next = arb = null;
	}

	Node(int a, int b) {
		coeff = a;
		pow = b;
	}
}