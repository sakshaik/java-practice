package com.test.euler;

import java.util.Scanner;

public class Problem26 {

	public static void main(String[] args) {
		System.out.println("Enter a no for which we need to get sum of all numbers divisible by 3 and 5 :");
		Scanner number = new Scanner(System.in);
		int num = number.nextInt();
		number.close();
		int result = 0;
		int longest = 0;
		for (int i = 2; i < num; i++) {
			int recurringNum = recurringNum(i);
			if (recurringNum > longest) {
				longest = recurringNum;
				result = i;
			}
		}
		System.out.println(result);
	}

	public static int recurringNum(int num) {
		int[] arr = new int[num + 1];
		int index = 1;
		int mod = 1;
		while (mod != 0 && arr[mod] == 0) {
			arr[mod] = index++;
			mod = mod * 10 % num;
		}
		if (mod == 0) {
			return 0;
		}
		return index - arr[mod];
	}

}