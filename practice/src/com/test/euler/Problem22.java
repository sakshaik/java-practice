package com.test.euler;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class Problem22 {

	public static void main(String[] args) throws IOException {
		ArrayList<String> names = new ArrayList<String>();
		BufferedReader numReader = new BufferedReader(new FileReader("C:\\Users\\sakshaik\\Downloads\\names.txt"));
		String line = "";

		while ((line = numReader.readLine()) != null) {
			String[] dataNames = line.split(",");

			for (String item : dataNames) {
				String newItem = item.substring(1, item.length() - 1);
				names.add(newItem);
			}
		}

		numReader.close();
		Collections.sort(names);
		
		int sum = 0;
		
		for (int i = 0; i < names.size(); i++) {
			int value = 0;
			for (int j = 0; j < names.get(i).length(); j++) {
				value += names.get(i).charAt(j) - 'A' + 1;
			}
			sum += value * (i + 1);
		}
		
		System.out.println(sum);
	}
}
