package com.mt.algorithm.algorithmstudy.basic;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

/**
 * @Description
 * @Author T
 * @Date 2022/9/1
 */
@SpringBootTest
public class PartNineDpTest {

    @Autowired
    private PartNineDp service;

    @Test
    public void leetCode322Test() {
        int[] testArr = {1, 2, 5};
        int result = service.leetCode322(testArr, 11);
        Assertions.assertEquals(3, result);
    }

    @Test
    public void leetCode354Test() {
        int[][] test = {{5, 4}, {6, 4}, {6, 7}, {2, 3}};
        int result = service.leetCode354(test);
        Assertions.assertEquals(3, result);
    }

    @Test
    public void leetCode1049Test() {
        int[] testArr = {1, 2, 4, 8, 16, 32, 64, 12, 25, 51};
        int result = service.leetCode1049(testArr);
        Assertions.assertEquals(1, result);
    }

    @Test
    public void leetCode494Test() {
        int[] testArr = {1, 1, 1, 1, 1};
        int result = service.leetCode494(testArr, 3);
        Assertions.assertEquals(5, result);
    }

    @Test
    public void leetCode518Test() {
        int[] testArr = {1, 2, 5};
        int result = service.leetCode518(5, testArr);
        Assertions.assertEquals(4, result);
    }

}