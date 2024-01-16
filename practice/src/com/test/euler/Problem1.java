package com.test.euler;

import java.util.Scanner;

// https://projecteuler.net/problem=1
public class Problem1 {

	public static void main(String[] args) {
		Scanner number = new Scanner(System.in);
		System.out.println("Enter a no for which we need to get sum of all numbers divisible by 3 and 5 :");
		int num = number.nextInt();
		System.out.println(getSumOfAllDivisible(num));
		number.close();
	}

	private static int getSumOfAllDivisible(int num) {
		int total = 0;
		for (int i = 0; i < num; i++) {
			if (i % 3 == 0 || i % 5 == 0)
				total += i;
		}
		return total;
	}

}
