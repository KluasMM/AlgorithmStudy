package com.mt.algorithm.algorithmstudy.basic;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

/**
 * @Description
 * @Author T
 * @Date 2022/7/31
 */
@SpringBootTest
public class PartFiveStackTest {

    @Autowired
    private PartFiveStack service;

    @Test
    void leetCode150Test() {
        String[] test = {"4", "13", "5", "/", "+"};
        int result = service.leetCode150(test);
        Assertions.assertEquals(6, result);
    }

    @Test
    void leetCode239Test() {
        int[] test1 = {1, 3, -1, -3, 5, 3, 6, 7};
        int test2 = 3;
        int[] result = service.leetCode239(test1, test2);
        Assertions.assertArrayEquals(new int[]{3, 3, 5, 5, 6, 7}, result);
    }
}
