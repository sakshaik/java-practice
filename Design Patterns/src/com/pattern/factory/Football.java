package com.pattern.factory;

public class Football implements Sport{

	@Override
	public void substitutes() {
		System.out.println("Football has 3 subs");
	}

}
