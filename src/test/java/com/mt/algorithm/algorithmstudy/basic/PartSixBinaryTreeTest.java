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
 * @Date 2022/8/1
 */
@SpringBootTest
public class PartSixBinaryTreeTest {

    @Autowired
    private PartSixBinaryTree service;

    @Test
    void leetCode515Test() {
        PartSixBinaryTree.TreeNode node33 = new PartSixBinaryTree.TreeNode(3);
        PartSixBinaryTree.TreeNode node35 = new PartSixBinaryTree.TreeNode(5);
        PartSixBinaryTree.TreeNode node23 = new PartSixBinaryTree.TreeNode(3, node35, node33);
        PartSixBinaryTree.TreeNode node39 = new PartSixBinaryTree.TreeNode(9);
        PartSixBinaryTree.TreeNode node22 = new PartSixBinaryTree.TreeNode(2, null, node39);
        PartSixBinaryTree.TreeNode root = new PartSixBinaryTree.TreeNode(1, node23, node22);

        List<Integer> result = service.leetCode515(root);
        Assertions.assertArrayEquals(Arrays.asList(1, 3, 9).toArray(), result.toArray());
    }

    @Test
    void leetCode104Test() {
        PartSixBinaryTree.TreeNode node33 = new PartSixBinaryTree.TreeNode(3);
        PartSixBinaryTree.TreeNode node35 = new PartSixBinaryTree.TreeNode(5);
        PartSixBinaryTree.TreeNode node23 = new PartSixBinaryTree.TreeNode(3, node35, node33);
        PartSixBinaryTree.TreeNode node39 = new PartSixBinaryTree.TreeNode(9);
        PartSixBinaryTree.TreeNode node22 = new PartSixBinaryTree.TreeNode(2, null, node39);
        PartSixBinaryTree.TreeNode root = new PartSixBinaryTree.TreeNode(1, node23, node22);

        int result = service.leetCode104(root);
        Assertions.assertEquals(3, result);
    }

    @Test
    void leetCode111Test() {
        PartSixBinaryTree.TreeNode node33 = new PartSixBinaryTree.TreeNode(3);
        PartSixBinaryTree.TreeNode node35 = new PartSixBinaryTree.TreeNode(5);
        PartSixBinaryTree.TreeNode node23 = new PartSixBinaryTree.TreeNode(3, node35, node33);
        PartSixBinaryTree.TreeNode node22 = new PartSixBinaryTree.TreeNode(2, null, null);
        PartSixBinaryTree.TreeNode root = new PartSixBinaryTree.TreeNode(1, node23, node22);

        int result = service.leetCode111(root);
        Assertions.assertEquals(2, result);
    }

    @Test
    void leetCode222Test() {
        PartSixBinaryTree.TreeNode node36 = new PartSixBinaryTree.TreeNode(6);
        PartSixBinaryTree.TreeNode node35 = new PartSixBinaryTree.TreeNode(5);
        PartSixBinaryTree.TreeNode node34 = new PartSixBinaryTree.TreeNode(4);
        PartSixBinaryTree.TreeNode node23 = new PartSixBinaryTree.TreeNode(3, node36, null);
        PartSixBinaryTree.TreeNode node22 = new PartSixBinaryTree.TreeNode(2, node34, node35);
        PartSixBinaryTree.TreeNode root = new PartSixBinaryTree.TreeNode(1, node22, node23);

        int result = service.leetCode222(root);
        Assertions.assertEquals(6, result);
    }

    @Test
    void leetCode98Test() {
        PartSixBinaryTree.TreeNode node22 = new PartSixBinaryTree.TreeNode(1, null, null);
        PartSixBinaryTree.TreeNode root = new PartSixBinaryTree.TreeNode(1, node22, null);

        boolean result = service.leetCode98(root);
        Assertions.assertFalse(result);
    }
}
