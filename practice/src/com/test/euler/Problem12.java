package com.test.euler;

public class Problem12 {

	public static int triangleNumber(int n) {
		int sum = 0;
		for (int i = 0; i <= n; i++)
			sum += i;
		return sum;
	}

	public static void main(String[] args) {

		long start = System.currentTimeMillis(); // start the stopwatch

		int j = 0; // j represents the jth triangle number
		int n = 0; // n represents the triangle number corresponding to j
		int numberOfDivisors = 0; // number of divisors for triangle number n

		while (numberOfDivisors <= 500) {

			// resets numberOfDivisors because it's now checking a new triangle number
			// and also sets n to be the next triangle number
			numberOfDivisors = 0;
			j++;
			n = triangleNumber(j);

			// for every number from 1 to the square root of this triangle number,
			// count the number of divisors
			for (int i = 1; i <= Math.sqrt(n); i++)
				if (n % i == 0)
					numberOfDivisors++;

			// 1 to the square root of the number holds exactly half of the divisors
			// so multiply it by 2 to include the other corresponding half
			numberOfDivisors *= 2;
		}

		long finish = System.currentTimeMillis(); // stop the stopwatch

		System.out.println(n);
		System.out.println("Time taken: " + (finish - start) + " milliseconds");
	}

}
