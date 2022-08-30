package com.mt.algorithm.algorithmstudy.basic;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.Arrays;
import java.util.List;

/**
 * @Description
 * @Author T
 * @Date 2022/8/17
 */
@SpringBootTest
public class PartSevenBacktrackTest {

    @Autowired
    private PartSevenBacktrack service;

    @Test
    public void leetCode93Test() {
        String test = "25525511135";
        List<String> result = service.leetCode93(test);
        Assertions.assertArrayEquals(
                Arrays.asList("255.255.11.135", "255.255.111.35").toArray(),
                result.toArray());
    }

    @Test
    public void leetCode78Test() {
        int[] test = {1, 2, 3};
        List<List<Integer>> result = service.leetCode78(test);
    }
}
