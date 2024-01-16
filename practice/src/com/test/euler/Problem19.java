package com.test.euler;

public class Problem19 {

	public static void main(String[] args) {
		System.out.println(new Problem19().run());
	}

	public String run() {
		int count = 0;
		for (int y = 1901; y <= 2000; y++) {
			for (int m = 1; m <= 12; m++) {
				if (dayOfWeek(y, m, 1) == 0) // Sunday
					count++;
			}
		}
		return Integer.toString(count);
	}

	private static int dayOfWeek(int year, int month, int day) {
		long m = mod((long) month - 3, 4800);
		long y = mod(year + m / 12, 400);
		m %= 12;
		return (int) ((y + y / 4 - y / 100 + (13 * m + 2) / 5 + day + 2) % 7);
	}

	private static long mod(long x, long y) {
		x %= y;
		if (y > 0 && x < 0 || y < 0 && x > 0)
			x += y;
		return x;
	}

}
