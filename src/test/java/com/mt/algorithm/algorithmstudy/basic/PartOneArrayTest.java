package com.mt.algorithm.algorithmstudy.basic;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

/**
 * @Description
 * @Author T
 * @Date 2022/7/17
 */
@SpringBootTest
public class PartOneArrayTest {

    @Autowired
    private PartOneArray partOneArray;

    @Test
    void leetCode844Test() {
        boolean result = partOneArray.leetCode844("ab##", "c#d#");

        Assertions.assertTrue(result);
    }

    @Test
    void leetCode977Test() {
        int[] test = {-4, -1, 0, 3, 10};
        int[] result = partOneArray.leetCode977(test);

        int[] expect = {0, 1, 9, 16, 100};
        Assertions.assertArrayEquals(expect, result);
    }

    @Test
    void leetCode904Test() {
        int[] test = {1, 0, 3, 4, 3};
        int result = partOneArray.leetCode904(test);

        Assertions.assertEquals(3, result);
    }

    @Test
    void leetCode76Test() {
        String s = "ADOBECODEBANC";
        String t = "ABC";
        String result = partOneArray.leetCode76(s, t);

        Assertions.assertEquals("BANC", result);
    }
}
