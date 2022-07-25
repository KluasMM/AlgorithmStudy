package com.mt.algorithm.algorithmstudy.basic;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * @Description
 * @Author T
 * @Date 2022/7/21
 */
@SpringBootTest
public class PartThreeHashTest {

    @Autowired
    private PartThreeHash service;

    @Test
    void leetCode49Test() {
        String[] test = {"eat", "tea", "tan", "ate", "nat", "bat"};
        List<List<String>> result = service.leetCode49(test);

        System.out.println(result);
    }

    @Test
    void leetCode438Test() {
        String test1 = "cbaebabacd";
        String test2 = "abc";
        List<Integer> result = service.leetCode438(test1, test2);
        Assertions.assertArrayEquals(Arrays.asList(0, 6).toArray(), result.toArray());
    }

    @Test
    void leetCode18Test() {
        int[] test1 = {0, 0, 0, 1000000000, 1000000000, 1000000000, 1000000000};
        int test2 = 1000000000;
        List<List<Integer>> result = service.leetCode18(test1, test2);
        Assertions.assertArrayEquals(Collections.emptyList().toArray(), result.toArray());
    }

}
