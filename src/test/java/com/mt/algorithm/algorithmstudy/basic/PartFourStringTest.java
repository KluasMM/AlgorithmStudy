package com.mt.algorithm.algorithmstudy.basic;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

/**
 * @Description
 * @Author T
 * @Date 2022/7/25
 */
@SpringBootTest
public class PartFourStringTest {

    @Autowired
    private PartFourString service;

    @Test
    void leetCode541Test() {
        String test1 = "abcdefg";
        int test2 = 2;
        String result = service.leetCode541(test1, test2);
        Assertions.assertEquals("bacdfeg", result);
    }

    @Test
    void leetCode151Test() {
        String test = "a";
        String result = service.leetCode151(test);
        Assertions.assertEquals("blue is sky the", result);
    }
}
