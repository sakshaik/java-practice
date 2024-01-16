package com.pattern.factory;

public class SportFactoryDemo {
	public static void main(String[] args) {
		SportFactory.getInstance().getSport("Football").substitutes();
		SportFactory.getInstance().getSport("Cricket").substitutes();
		SportFactory.getInstance().getSport("Rugby").substitutes();
		SportFactory.getInstance().getSport("Basketball").substitutes();
		SportFactory.getInstance().getSport(null).substitutes();	
	}
}
